"""
Feature definitions. Each feature is (name, kind, description).
kind: rating (int 0-4) | count (int) | flag (bool) | frac (float 0-1)

Design rules (to stop the downstream model memorising a particular recording):
  * every feature is a behaviour-level metric with an anchored scale, so it means the same
    thing in any class by any tutor;
  * no feature refers to appearance/identity/content specifics (colours, names, subject, room);
  * the prompt never reveals the class label and never asks for an overall quality judgement;
  * only fixed-schema numbers go to the classifier - the free-text 'evidence' is for humans.
"""
from pydantic import create_model, Field

R = "rating"; C = "count"; F = "flag"; P = "frac"

TEXT_FEATURES = [
    ("explanation_clarity", R, "Tutor explanations are clear, sequenced and age-appropriate. 0=confusing/none, 2=adequate, 4=consistently clear, with examples or step-by-step reasoning."),
    ("understanding_checking", R, "Tutor probes understanding (asks learners to explain, try, or redo; follows up on wrong answers). 0=never, 2=occasionally, 4=systematic."),
    ("learner_participation", R, "Learners actively respond, attempt answers or ask questions. 0=no learner speech, 2=some short replies, 4=sustained, varied contributions."),
    ("individual_attention", R, "Tutor invites or addresses learners one by one (calls on each, personalised comments). 0=none, 4=every learner repeatedly."),
    ("encouragement_warmth", R, "Positive, friendly, encouraging language; specific praise. 0=cold/none, 2=neutral-polite, 4=consistently warm and motivating."),
    ("feedback_quality", R, "Corrections and feedback are constructive and specific. 0=none/harsh, 4=precise, kind and actionable."),
    ("structure_pacing", R, "Segment has a purpose, logical sequence, good pace, little dead time. 0=aimless/stalled, 4=tightly structured."),
    ("tutor_energy_language", R, "Lively, enthusiastic wording (varied phrasing, exclamations, humour) vs flat/monotone wording. 0=flat, 4=very lively."),
    ("classroom_management", R, "Keeps attention, handles interruptions/off-task behaviour smoothly. 0=loses control, 4=effortless."),
    ("tech_disruption", R, "Evidence in the transcript of audio/connection problems (can you hear me, repeating, lag). 0=none, 4=severe."),
    ("off_task_share", R, "Share of the segment spent on unrelated talk or waiting. 0=none, 4=most of it."),
    ("tutor_questions", C, "Number of questions the tutor asks learners."),
    ("open_questions", C, "Number of those tutor questions that need an explanation or reasoning, not a one-word answer."),
    ("learners_addressed_individually", C, "Number of times the tutor directs a question/comment to one specific learner."),
    ("praise_instances", C, "Number of distinct praise or encouragement statements by the tutor."),
    ("comprehension_checks", C, "Number of explicit checks such as 'does that make sense?', 'any doubts?', 'can you repeat it?'."),
    ("greeting_or_rapport", F, "Tutor greets learners or builds rapport (asks how they are, light conversation) in this segment."),
    ("learner_introductions", F, "Learners (esp. new ones) are given time to introduce themselves."),
    ("objectives_stated", F, "Tutor states what will be learned/done in this lesson or activity."),
    ("recap_or_closing", F, "Tutor summarises, recaps, or formally closes the lesson in this segment."),
    ("practice_or_homework_assigned", F, "Tutor assigns practice, homework or a follow-up task."),
]

VIDEO_FEATURES = [
    ("tutor_on_camera_frac", P, "Fraction of the segment in which the tutor is visible on camera (0.0-1.0)."),
    ("tutor_face_clarity", R, "Tutor's face is fully visible, well framed and not cut off. 0=not visible, 2=partially, 4=clear and well framed throughout."),
    ("lighting_quality", R, "Lighting on the tutor is adequate and even. 0=too dark/blown out, 4=bright and even."),
    ("video_stability_sharpness", R, "Picture is sharp and stable without freezing or heavy blur. 0=unwatchable, 4=crisp and stable."),
    ("backdrop_neatness", R, "Tutor's surroundings look tidy, uncluttered and appropriate for a lesson. 0=distracting/messy, 4=clean and professional."),
    ("audio_clarity", R, "Tutor's voice is clearly intelligible. 0=unintelligible, 4=crisp."),
    ("background_noise", R, "Intrusive noise or echo. 0=none, 4=severe."),
    ("tutor_expressiveness", R, "Facial expression: smiling, pleasant, engaged. 0=blank/irritated, 4=consistently warm and expressive."),
    ("tutor_vocal_energy", R, "Voice has varied pitch/pace/volume and enthusiasm. 0=monotone/flat, 4=highly dynamic."),
    ("gesture_use", R, "Purposeful gestures/pointing/props that support explanation. 0=none, 4=frequent and purposeful."),
    ("eye_contact_camera", R, "Tutor looks at the camera/learners rather than away. 0=rarely, 4=consistently."),
    ("visual_aid_use", R, "Use of slides, whiteboard, screen share, props, or videos to teach. 0=none, 4=rich and well integrated."),
    ("interactive_activity", R, "Use of interactive activities (puzzles, games, polls, worksheets, learner turn-taking tasks). 0=none, 4=central to segment."),
    ("learner_visible_engagement", R, "If learners are visible: attentive and participating. 0=learners not visible or disengaged, 4=clearly engaged."),
    ("tech_disruption", R, "Freezes, dropouts, audio cut-offs, camera switching off. 0=none, 4=severe."),
    ("tutor_speaking_frac", P, "Fraction of the segment in which the tutor is speaking (0.0-1.0)."),
    ("learner_speaking_frac", P, "Fraction of the segment in which any learner is speaking (0.0-1.0)."),
    ("screen_share_active", F, "A screen or document is being shared."),
    ("learners_visible", F, "At least one learner's video is visible."),
    ("tutor_audio_cutoffs", F, "The tutor's audio cuts out or is badly distorted at least once."),
]

_T = {R: int, C: int, F: bool, P: float}


def make_model(name, spec):
    fields = {"evidence": (str, Field(description="One short sentence citing observable behaviours behind the ratings. No names."))}
    for n, k, d in spec:
        fields[n] = (_T[k], Field(description=d))
    return create_model(name, **fields)


def clean(spec, d):
    """Clamp model output into valid ranges -> flat numeric dict."""
    out = {}
    for n, k, _ in spec:
        v = d.get(n, 0)
        try:
            if k == R: v = min(4, max(0, int(round(float(v)))))
            elif k == C: v = min(50, max(0, int(round(float(v)))))
            elif k == F: v = int(bool(v))
            else: v = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            v = 0
        out[n] = v
    return out


RULES = """You are a careful classroom-observation coder. You will code ONE short segment of an online \
tutoring session for children. Rules:
- Code only behaviours that are observable in THIS segment. Do not guess about the rest of the lesson and do \
not give an overall quality verdict.
- Do not use or output any person's name or any identifying detail.
- Rating fields are integers 0-4 using the anchors in each field description \
(0=absent/very poor, 1=weak, 2=adequate, 3=good, 4=exemplary). Use the full range; do not default to the middle.
- Count fields are integer counts of observed instances in this segment. Flag fields are true only if clearly observed. \
Fraction fields are floats 0.0-1.0.
- If something cannot be observed, give 0 / false."""
