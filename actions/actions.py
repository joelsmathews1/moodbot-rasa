"""
MoodBot – actions.py
=====================
Custom actions for emotion detection, empathetic response selection,
contextual coping suggestions, crisis handling, and journaling prompts.

Architecture:
  • EmotionClassifier  – rule-based + keyword scoring hybrid
  • IntensityScorer    – mild / moderate / severe based on linguistic cues
  • Custom Rasa actions that compose the above
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted

logger = logging.getLogger(__name__)


# ==============================================================
# Emotion Classifier (Rule-based + keyword scoring)
# ==============================================================

# Each emotion maps to a weighted keyword list.
# Tuples are (keyword_pattern, weight).
EMOTION_LEXICON: Dict[str, List[Tuple[str, float]]] = {
    "sad": [
        (r"\bsad\b", 1.0),
        (r"\bdepressed?\b", 1.2),
        (r"\bdown\b", 0.8),
        (r"\blonely\b", 1.0),
        (r"\bgrief\b", 1.1),
        (r"\bgriev", 1.1),
        (r"\bhopeless\b", 1.3),
        (r"\bempty\b", 0.9),
        (r"\bheartbrok", 1.1),
        (r"\bmiser", 1.0),
        (r"\bdevastated?\b", 1.2),
        (r"\bcrying\b", 1.0),
        (r"\bcried\b", 1.0),
        (r"\bpointless\b", 1.0),
        (r"\bnumb\b", 0.7),
        (r"\bunhappy\b", 0.9),
        (r"\bbluе?\b", 0.6),
        (r"\bsuffer", 0.9),
        (r"\bdisconnect", 0.7),
        (r"\bmiss (someone|him|her|them)\b", 0.8),
    ],
    "stressed": [
        (r"\bstress(ed)?\b", 1.2),
        (r"\bburnout\b", 1.3),
        (r"\bburnt? out\b", 1.3),
        (r"\bpressure\b", 1.0),
        (r"\bexhausted?\b", 1.0),
        (r"\boverload", 1.1),
        (r"\bdeadline", 0.9),
        (r"\btoo much (to do|on my plate)\b", 1.2),
        (r"\bcan'?t keep up\b", 1.1),
        (r"\brunning on empty\b", 1.0),
        (r"\bno (time|break|rest)\b", 0.8),
        (r"\bworn out\b", 0.9),
        (r"\bdrained\b", 1.0),
        (r"\bno energy\b", 0.8),
    ],
    "anxious": [
        (r"\banxi(ous|ety)\b", 1.3),
        (r"\bworr(ied|y|ying)\b", 1.0),
        (r"\bnervous\b", 1.0),
        (r"\bpanic(k(ing|ed))?\b", 1.3),
        (r"\bscared\b", 0.9),
        (r"\bfear(ful)?\b", 0.9),
        (r"\bon edge\b", 1.0),
        (r"\brace?(ing)? (thoughts?|mind)\b", 1.1),
        (r"\bcatastrophi", 1.2),
        (r"\bdread\b", 1.0),
        (r"\bparanoi", 1.2),
        (r"\bcan'?t relax\b", 0.9),
        (r"\buneasy\b", 0.8),
        (r"\bapprehen", 0.9),
    ],
    "overwhelmed": [
        (r"\boverwhelmе?d?\b", 1.3),
        (r"\btoo much\b", 1.0),
        (r"\bburied\b", 0.9),
        (r"\bswamped\b", 1.0),
        (r"\bdrown(ing)?\b", 1.1),
        (r"\bcan'?t cope\b", 1.1),
        (r"\bpiling up\b", 1.0),
        (r"\bpile up\b", 1.0),
        (r"\bparalyz", 1.0),
        (r"\bwhere (to start|do i begin)\b", 0.9),
        (r"\bno way out\b", 1.0),
        (r"\beveryth(ing|s) is (too much|falling apart)\b", 1.2),
    ],
    "angry": [
        (r"\bangr(y|ier)\b", 1.2),
        (r"\bfurious\b", 1.3),
        (r"\brage\b", 1.3),
        (r"\blivid\b", 1.3),
        (r"\bfrustr(ated?|ation)\b", 1.1),
        (r"\birrit(ated?|ation)\b", 1.0),
        (r"\bpissed\b", 1.1),
        (r"\bfed up\b", 1.0),
        (r"\bresentful?\b", 1.0),
        (r"\bbetrayed?\b", 1.1),
        (r"\bsnapp(ed|ing)\b", 0.9),
        (r"\bbitter(ness)?\b", 1.0),
        (r"\benraged?\b", 1.3),
        (r"\bnot fair\b", 0.7),
        (r"\bthis is (ridiculous|unacceptable|unfair)\b", 0.9),
    ],
    "happy": [
        (r"\bhappy\b", 1.2),
        (r"\bjoy(ful)?\b", 1.2),
        (r"\bgreat\b", 0.8),
        (r"\bexcited?\b", 1.0),
        (r"\bpositive\b", 0.7),
        (r"\bthankful\b", 0.8),
        (r"\bgrateful?\b", 0.8),
        (r"\bproud\b", 0.9),
        (r"\bcontent(ment)?\b", 0.9),
        (r"\bhopeful?\b", 0.8),
        (r"\bgood day\b", 0.7),
        (r"\baccomplish", 0.8),
        (r"\bbreakthrough\b", 0.8),
        (r"\blooking up\b", 0.7),
        (r"\blight(er)?\b", 0.6),
    ],
}

INTENSITY_AMPLIFIERS = [
    (r"\bvery\b", 0.3),
    (r"\bextremely\b", 0.5),
    (r"\bso\b", 0.2),
    (r"\breally\b", 0.2),
    (r"\bincredibly\b", 0.5),
    (r"\bcompletely\b", 0.4),
    (r"\babsolutely\b", 0.4),
    (r"\btotally\b", 0.3),
    (r"\bcan'?t (take|handle|cope|stand)\b", 0.5),
    (r"\bI (hate|loathe|despise)\b", 0.4),
    (r"\bnever felt\b", 0.3),
    (r"\bworse than ever\b", 0.5),
]

CRISIS_PATTERNS = [
    r"\b(kill|end|hurt)\s*(my)?self\b",
    r"\b(don'?t|do not)\s*want\s*(to)?\s*(live|be here|exist|go on)\b",
    r"\bsuicid(e|al|ally)\b",
    r"\bself[- ]?harm(ing)?\b",
    r"\bwant (to die|to disappear|to end it)\b",
    r"\bnot worth living\b",
    r"\beveryone (would be )?better off without me\b",
    r"\bmaking a plan (to end|to kill)\b",
    r"\bI'?m (going to|planning to) (kill|hurt|end)\b",
]


@dataclass
class EmotionResult:
    emotion: str
    score: float
    intensity: str  # mild | moderate | severe
    all_scores: Dict[str, float] = field(default_factory=dict)
    crisis_detected: bool = False


class EmotionClassifier:
    """
    Hybrid rule-based emotion classifier.

    Scoring:
      1. Keyword matching with weights from EMOTION_LEXICON
      2. Amplifier boost applied to base score
      3. Intensity bucketed: mild (<1.5), moderate (1.5-3.0), severe (>3.0)

    Falls back to 'unknown' if no emotion scores above threshold.
    """

    UNKNOWN_THRESHOLD = 0.5

    @staticmethod
    def _is_crisis(text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in CRISIS_PATTERNS)

    @staticmethod
    def _amplifier_boost(text: str) -> float:
        text_lower = text.lower()
        boost = 0.0
        for pattern, weight in INTENSITY_AMPLIFIERS:
            if re.search(pattern, text_lower):
                boost += weight
        return min(boost, 1.0)  # cap at 1.0

    @staticmethod
    def _score_emotion(text: str, patterns: List[Tuple[str, float]]) -> float:
        text_lower = text.lower()
        score = 0.0
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower)
            score += len(matches) * weight
        return score

    @classmethod
    def classify(cls, text: str) -> EmotionResult:
        if not text or not text.strip():
            return EmotionResult("unknown", 0.0, "mild")

        crisis = cls._is_crisis(text)
        boost = cls._amplifier_boost(text)

        scores: Dict[str, float] = {}
        for emotion, patterns in EMOTION_LEXICON.items():
            raw = cls._score_emotion(text, patterns)
            scores[emotion] = raw + (boost * raw * 0.5) if raw > 0 else 0.0

        if not any(v > cls.UNKNOWN_THRESHOLD for v in scores.values()):
            return EmotionResult("unknown", 0.0, "mild", scores, crisis)

        top_emotion = max(scores, key=scores.get)
        top_score = scores[top_emotion]

        # Intensity bucketing
        effective_score = top_score + boost
        if effective_score < 1.5:
            intensity = "mild"
        elif effective_score < 3.0:
            intensity = "moderate"
        else:
            intensity = "severe"

        return EmotionResult(top_emotion, top_score, intensity, scores, crisis)


# ==============================================================
# Coping response tables
# ==============================================================

BREATHING_RESPONSES = {
    "anxious": "utter_breathing_478",       # 4-7-8 best for anxiety
    "stressed": "utter_breathing_box",      # Box breathing for stress
    "overwhelmed": "utter_breathing_box",
    "angry": "utter_breathing_box",
    "sad": "utter_breathing_478",
    "default": "utter_breathing_box",
}

JOURNALING_RESPONSES = {
    "sad": "utter_journaling_prompt_sad",
    "stressed": "utter_journaling_prompt_stressed",
    "anxious": "utter_journaling_prompt_anxious",
    "overwhelmed": "utter_journaling_prompt_stressed",
    "angry": "utter_journaling_prompt_angry",
    "default": "utter_journaling_prompt_generic",
}

EMPATHY_RESPONSES = {
    "sad": "utter_empathy_sad",
    "stressed": "utter_empathy_stressed",
    "anxious": "utter_empathy_anxious",
    "overwhelmed": "utter_empathy_overwhelmed",
    "angry": "utter_empathy_angry",
    "happy": "utter_empathy_happy",
    "neutral": "utter_empathy_neutral",
    "unknown": "utter_empathy_unknown",
}


# ==============================================================
# Custom Actions
# ==============================================================

class ActionDetectEmotion(Action):
    """
    Runs the EmotionClassifier against the last user message
    and stores results in slots.
    """

    def name(self) -> Text:
        return "action_detect_emotion"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Collect recent user messages for better context
        recent_messages = []
        for event in tracker.events[-10:]:
            if event.get("event") == "user" and event.get("text"):
                recent_messages.append(event["text"])

        combined_text = " ".join(recent_messages)
        result = EmotionClassifier.classify(combined_text)

        logger.info(
            f"[EmotionClassifier] emotion={result.emotion}, "
            f"intensity={result.intensity}, crisis={result.crisis_detected}, "
            f"scores={result.all_scores}"
        )

        events = [
            SlotSet("detected_emotion", result.emotion),
            SlotSet("emotion_intensity", result.intensity),
        ]

        # If crisis detected, flag it immediately (rule will also catch it,
        # but belt-and-suspenders for custom fallback contexts)
        if result.crisis_detected:
            logger.warning("[CRISIS DETECTED] Flagged from action_detect_emotion")

        return events


class ActionRespondWithEmpathy(Action):
    """
    Dispatches an emotion-matched empathetic response.
    Considers intensity to vary language if severe.
    """

    def name(self) -> Text:
        return "action_respond_with_empathy"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        emotion = tracker.get_slot("detected_emotion") or "unknown"
        intensity = tracker.get_slot("emotion_intensity") or "mild"

        response_key = EMPATHY_RESPONSES.get(emotion, "utter_empathy_unknown")
        dispatcher.utter_message(response=response_key)

        # For severe intensity, add a follow-up offering
        if intensity == "severe" and emotion not in ("happy",):
            dispatcher.utter_message(
                text=(
                    "It sounds like things are really difficult right now. "
                    "Would it help to try a quick grounding exercise, or would you "
                    "rather just keep talking?"
                )
            )

        # Always offer the coping menu after empathy for non-happy states
        if emotion not in ("happy", "neutral", "unknown"):
            dispatcher.utter_message(response="utter_offer_coping_menu")

        return []


class ActionOfferContextualCoping(Action):
    """
    Provides a breathing exercise matched to the detected emotion.
    Falls back to box breathing if emotion is unknown.
    """

    def name(self) -> Text:
        return "action_offer_contextual_coping"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        emotion = tracker.get_slot("detected_emotion") or "default"
        response_key = BREATHING_RESPONSES.get(emotion, BREATHING_RESPONSES["default"])
        dispatcher.utter_message(response=response_key)

        return []


class ActionProvideJournalingPrompt(Action):
    """
    Returns a journaling prompt matched to the current detected emotion.
    """

    def name(self) -> Text:
        return "action_provide_journaling_prompt"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        emotion = tracker.get_slot("detected_emotion") or "default"
        response_key = JOURNALING_RESPONSES.get(emotion, JOURNALING_RESPONSES["default"])
        dispatcher.utter_message(response=response_key)

        return []


class ActionHandleCrisis(Action):
    """
    SAFETY CRITICAL ACTION.

    Fires on crisis_signal intent (also guarded by a Rule).
    Provides crisis resources without simulating a therapist.
    Logs a warning for monitoring/audit trail.

    In production: integrate with your alerting system here
    (e.g., PagerDuty, Slack ops channel webhook, etc.)
    """

    def name(self) -> Text:
        return "action_handle_crisis"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_id = tracker.sender_id
        latest_message = tracker.latest_message.get("text", "")

        # AUDIT LOG — in production, also write to a secure log store
        logger.critical(
            f"[CRISIS ESCALATION] sender_id={user_id} | "
            f"message='{latest_message[:80]}...'"
        )

        # TODO (production): send alert to ops channel
        # self._alert_ops_channel(user_id, latest_message)

        dispatcher.utter_message(response="utter_crisis_response")
        return [SlotSet("detected_emotion", "unknown")]

    @staticmethod
    def _alert_ops_channel(user_id: str, message: str) -> None:
        """
        Stub for production alerting integration.
        Example: POST to a Slack ops webhook.
        """
        # import requests
        # requests.post(OPS_WEBHOOK_URL, json={
        #     "text": f":rotating_light: Crisis signal from {user_id}: {message[:200]}"
        # })
        pass