# VidNotes — Feature Suggestions

Analysis and prioritized roadmap for turning VidNotes from a working tool into a
valuable product. The app already covers: JWT auth with email verification,
background transcription, SQLite persistence, per-video RAG chat, TTS, STT,
categories, and a Datastar-driven UI.

The tiers below are ordered roughly by effort/value.

## Tier 1 — Quick wins (days, high value)

1. **Export formats (SRT / VTT / TXT / Markdown / PDF)**
   Students and creators republish transcripts. Timestamped segments are already
   stored, so SRT/VTT export is a small amount of code. Biggest easy win for
   real users.

2. **AI summary + key takeaways**
   One Gemini call per transcript: TL;DR, bullet points, and auto-generated
   chapters/table of contents (from timestamped segments). Turns a transcript
   into notes people actually read.

3. **Full-text search with jump-to-timestamp**
   Add SQLite FTS5 over `videos`/`segments` so users search across their library
   and jump straight to the moment. Every segment already carries `start`/`end`.

4. **Transcript ↔ audio sync player**
   Click a timestamp to play that part of the video (embed the YouTube player
   with `start=`), highlight segments while audio plays. Easy and high-impact.

5. **Docker + docker-compose**
   One-command deploy (app + FFmpeg). Removes the biggest friction for
   self-hosting.

## Tier 2 — Differentiators (the "killer features")

6. **Cross-video / library-wide chat**
   Today RAG is per-video (`app/services/chat.py`). Let users ask questions
   across all their transcripts at once (retrieve over all chunks). This turns
   their library into a searchable knowledge base — the feature that makes
   VidNotes worth paying for vs. basic transcript tools.

7. **Playlist / batch transcription**
   Loop over all video IDs in a `list=` URL. Natural fit for the existing
   background pipeline.

8. **Speaker diarization** (via WhisperX / pyannote)
   Identify "Speaker 1/2" in transcripts. Huge for podcasts and meetings; the
   segment table already supports it.

9. **Translation**
   Gemini translate a transcript or chat cross-lingually. Broadens the market.

10. **Public sharing links**
    Generate a read-only `?token=` link to a transcript (readable without an
    account). Drives organic growth.

## Tier 3 — Scale & reliability

11. **Job queue (Redis/Celery or DB-backed)**
    Replace in-process `asyncio` tasks with a durable queue so transcription
    survives restarts and scales to multiple workers.

12. **Dedicated vector store**
    SQLite + numpy BLOBs caps retrieval at tens of thousands of chunks. Swap to
    `sqlite-vec` or ChromaDB as the library grows.

13. **Metadata enrichment**
    Persist thumbnail, channel avatar, publish date, view count. Richer cards on
    the dashboard.

14. **Rate limiting + usage quotas per user**
    Protect the transcribe/STT endpoints before opening to paying users.

## Tier 4 — Monetization (if going SaaS)

15. **Stripe subscription tiers**
    Free (X minutes/month) vs. paid (unlimited, summaries, cross-video chat).
    Usage data already tracked supports this.

16. **Admin panel**
    User/video stats, storage reporting.

## Top recommendation

Start with **#1 (export)**, **#2 (summary/chapters)**, and **#6 (library-wide
chat)** first — they are the difference between "a tool I use" and "a product
people tell others about."
