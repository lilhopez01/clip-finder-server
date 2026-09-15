// Clip Finder server
// Handles: video upload -> transcription -> AI moment-finding -> cutting -> download

import express from "express";
import multer from "multer";
import cors from "cors";
import { v4 as uuid } from "uuid";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import ffmpegPath from "ffmpeg-static";
import ffmpeg from "fluent-ffmpeg";

ffmpeg.setFfmpegPath(ffmpegPath);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const UPLOAD_DIR = path.join(__dirname, "tmp");
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: UPLOAD_DIR, limits: { fileSize: 5 * 1024 * 1024 * 1024 } }); // 5GB cap

// In-memory job store. Fine for personal use -- resets if the server restarts.
const jobs = {}; // job_id -> { status, title, sourcePath, sentences, clips: [{...meta, filePath}] }

const ASSEMBLYAI_KEY = process.env.ASSEMBLYAI_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// ---------- 1. Start a job: upload video, kick off processing in background ----------
app.post("/jobs", upload.single("video"), async (req, res) => {
  const jobId = uuid();
  const title = req.body.title || "Untitled";
  const sourcePath = req.file.path;

  jobs[jobId] = { status: "uploaded", title, sourcePath, clips: [], error: null };
  res.json({ job_id: jobId });

  processJob(jobId).catch((err) => {
    console.error("Job failed:", err);
    jobs[jobId].status = "error";
    jobs[jobId].error = String(err.message || err);
  });
});

// ---------- 2. Check job status / get results ----------
app.get("/jobs/:id", (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json({
    status: job.status,
    title: job.title,
    error: job.error,
    clips: job.clips.map((c, i) => ({
      index: i,
      start_ms: c.start_ms,
      end_ms: c.end_ms,
      category: c.category,
      explanation: c.explanation,
      tips: c.tips,
      download_url: `/jobs/${req.params.id}/clips/${i}/download`,
    })),
  });
});

// ---------- 3. Download a specific cut clip ----------
app.get("/jobs/:id/clips/:index/download", (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.status(404).send("Job not found");
  const clip = job.clips[req.params.index];
  if (!clip || !fs.existsSync(clip.filePath)) return res.status(404).send("Clip not found");
  res.download(clip.filePath, `${job.title}-clip-${Number(req.params.index) + 1}.mp4`);
});

app.get("/", (req, res) => res.send("Clip Finder server is running."));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));

// ================= Processing pipeline =================

async function processJob(jobId) {
  const job = jobs[jobId];

  // ---- Transcribe ----
  job.status = "transcribing";
  const audioUrl = await uploadToAssemblyAI(job.sourcePath);
  const transcriptId = await startTranscript(audioUrl);
  const sentences = await pollTranscript(transcriptId);
  job.sentences = sentences;

  // ---- Find interesting moments ----
  job.status = "analyzing";
  const clips = await findClips(sentences);

  // ---- Cut each clip ----
  job.status = "cutting";
  const clipDir = path.join(UPLOAD_DIR, jobId);
  fs.mkdirSync(clipDir, { recursive: true });

  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    const outPath = path.join(clipDir, `clip_${i}.mp4`);
    await cutClip(job.sourcePath, outPath, c.start_ms, c.end_ms);
    job.clips.push({ ...c, filePath: outPath });
  }

  // ---- Clean up the original full-length upload; we only need the clips now ----
  fs.unlink(job.sourcePath, () => {});

  job.status = "done";

  // ---- Auto-delete clips after 2 hours to avoid filling up disk on the free tier ----
  setTimeout(() => {
    fs.rm(clipDir, { recursive: true, force: true }, () => {});
    delete jobs[jobId];
  }, 2 * 60 * 60 * 1000);
}

async function uploadToAssemblyAI(filePath) {
  const stream = fs.createReadStream(filePath);
  const res = await fetch("https://api.assemblyai.com/v2/upload", {
    method: "POST",
    headers: { authorization: ASSEMBLYAI_KEY },
    body: stream,
    duplex: "half",
  });
  const data = await res.json();
  if (!res.ok) throw new Error("AssemblyAI upload failed: " + JSON.stringify(data));
  return data.upload_url;
}

async function startTranscript(audioUrl) {
  const res = await fetch("https://api.assemblyai.com/v2/transcript", {
    method: "POST",
    headers: { authorization: ASSEMBLYAI_KEY, "content-type": "application/json" },
    body: JSON.stringify({ audio_url: audioUrl }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error("AssemblyAI transcript start failed: " + JSON.stringify(data));
  return data.id;
}

async function pollTranscript(transcriptId) {
  while (true) {
    const res = await fetch(`https://api.assemblyai.com/v2/transcript/${transcriptId}`, {
      headers: { authorization: ASSEMBLYAI_KEY },
    });
    const data = await res.json();
    if (data.status === "completed") break;
    if (data.status === "error") throw new Error("Transcription failed: " + data.error);
    await new Promise((r) => setTimeout(r, 8000));
  }
  const sentRes = await fetch(`https://api.assemblyai.com/v2/transcript/${transcriptId}/sentences`, {
    headers: { authorization: ASSEMBLYAI_KEY },
  });
  const sentData = await sentRes.json();
  return (sentData.sentences || []).map((s) => ({ text: s.text, start: s.start, end: s.end }));
}

const RESPONSE_SCHEMA = {
  type: "OBJECT",
  properties: {
    clips: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          start_sentence: { type: "INTEGER" },
          end_sentence: { type: "INTEGER" },
          category: { type: "STRING" },
          explanation: { type: "STRING" },
          tips: { type: "ARRAY", items: { type: "STRING" } },
        },
        required: ["start_sentence", "end_sentence", "category", "explanation", "tips"],
      },
    },
  },
  required: ["clips"],
};

async function findClips(sentences) {
  const transcriptText = sentences.map((s, i) => `[${i}] ${s.text}`).join("\n");

  const prompt = `You are helping a video creator find the most interesting, clip-worthy moments in a long video transcript.

The transcript is a numbered list of sentences. Find distinct moments that are genuinely interesting: a story, a strong opinion, a surprising reaction, a useful piece of information, or a moment of real tension or emotion. Each moment must be a COMPLETE, self-contained beat with a clear beginning and end -- never cut off mid-thought. It should work as a standalone short clip, roughly 20 seconds to 2 minutes long.

Only pick moments that are truly worth clipping -- do not force a fixed number. Skip filler, small talk, and anything that doesn't stand on its own as a complete moment.

For each moment, return:
- start_sentence and end_sentence: sentence index range (inclusive) covering the whole moment, start to finish, making sure it begins and ends at natural boundaries
- category: a short 1-2 word tag (e.g. "Story", "Insight", "Reaction", "Hot take")
- explanation: 1-2 sentences on why this moment is interesting and worth watching
- tips: 2-4 short, concrete editing tips specific to this clip, for someone polishing it in CapCut

Transcript:
${transcriptText}`;

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=${GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { responseMimeType: "application/json", responseSchema: RESPONSE_SCHEMA },
      }),
    }
  );
  const data = await res.json();
  if (!res.ok) throw new Error("Gemini request failed: " + JSON.stringify(data));

  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
  const parsed = JSON.parse(raw);
  const rawClips = parsed.clips || [];

  return rawClips
    .filter((c) => sentences[c.start_sentence] && sentences[c.end_sentence])
    .map((c) => ({
      start_ms: sentences[c.start_sentence].start,
      end_ms: sentences[c.end_sentence].end,
      category: c.category,
      explanation: c.explanation,
      tips: c.tips,
    }));
}

function cutClip(sourcePath, outPath, startMs, endMs) {
  return new Promise((resolve, reject) => {
    ffmpeg(sourcePath)
      .setStartTime(startMs / 1000)
      .setDuration((endMs - startMs) / 1000)
      .outputOptions(["-c copy"])
      .save(outPath)
      .on("end", resolve)
      .on("error", reject);
  });
}
