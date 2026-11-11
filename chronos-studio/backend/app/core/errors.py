"""
core/errors.py — the app's entire error vocabulary.

Design rule (CRAFT R0.9): every error that reaches a user is (a) one plain-English
sentence, (b) one suggested fix, and (c) a technical detail that is collapsed by
default. A raw stack trace is never the primary message.

Every exception here carries `code`, `plain`, `fix`, `technical` and `http_status`,
so the API layer can serialize it uniformly and the frontend can render it without
special-casing.
"""

from __future__ import annotations

import traceback
from typing import Any


class ChronosError(Exception):
    """Base class for every expected failure in Chronos Studio."""

    code: str = "chronos.error"
    http_status: int = 400
    plain: str = "Something went wrong, but the analysis is safe."
    fix: str = "Try again. If it keeps happening, download the report archive and include it in your bug report."

    def __init__(
        self,
        plain: str | None = None,
        fix: str | None = None,
        technical: str | None = None,
        context: dict[str, Any] | None = None,
        code: str | None = None,
    ) -> None:
        if code:
            self.code = code
        self.plain = plain or self.plain
        self.fix = fix or self.fix
        self.technical = technical or ""
        self.context = context or {}
        super().__init__(self.plain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "plain": self.plain,
            "fix": self.fix,
            "technical": self.technical,
            "context": self.context,
        }


# --------------------------------------------------------------------------- #
# Data / ingestion
# --------------------------------------------------------------------------- #
class UnsupportedFormatError(ChronosError):
    code = "data.unsupported_format"
    http_status = 415
    plain = "That file type is not supported yet."
    fix = "Convert it to CSV or Excel (.xlsx) and upload again."


class FileTooLargeError(ChronosError):
    code = "data.too_large"
    http_status = 413
    plain = "That file is too large for the browser version."
    fix = "Upload a shorter date range, or split the file into several series."


class EmptyDatasetError(ChronosError):
    code = "data.empty"
    http_status = 422
    plain = "The file loaded, but there are no data rows in it."
    fix = "Check that the sheet or delimiter is correct and that row 1 is a header."


class TimeIndexError(ChronosError):
    """Raised when no usable time index can be constructed."""

    code = "data.time_index"
    http_status = 422
    plain = "We could not work out which column holds the dates (or times)."
    fix = "Pick the time column yourself in the Column mapping step."


class FrequencyInferenceError(ChronosError):
    code = "data.frequency"
    http_status = 422
    plain = "We could not tell how often your observations are recorded."
    fix = "Choose the frequency (daily, monthly, quarterly, ...) in the Column mapping step."


class IrregularSpacingError(ChronosError):
    code = "data.irregular"
    http_status = 422
    plain = "The gaps between your dates are not equal, so this is not a regular time series."
    fix = "Aggregate to a coarser regular frequency, or add the missing dates with blank values."


class ColumnMappingError(ChronosError):
    code = "data.column_mapping"
    http_status = 422
    plain = "We could not match the columns you chose to the data."
    fix = "Re-open Column mapping and pick the time column and value column again."


class NonNumericSeriesError(ChronosError):
    code = "data.non_numeric"
    http_status = 422
    plain = "The column you chose holds text, not numbers."
    fix = "Pick a numeric column, or clean the text (remove currency symbols, commas, and '#N/A')."


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class NonPositiveDataError(ChronosError):
    code = "prep.non_positive"
    http_status = 422
    plain = "A log or multiplicative transform cannot be applied because the series contains zero or negative values."
    fix = "Use an additive method instead, or shift the data (log(y + c)) and record the constant c."


class InsufficientDataError(ChronosError):
    code = "prep.insufficient_data"
    http_status = 422
    plain = "There are not enough observations for the step you asked for."
    fix = "Use a simpler model, or extend the series to at least two full seasonal cycles."


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
class ModelNotFoundError(ChronosError):
    code = "model.not_found"
    http_status = 404
    plain = "That model is not in this build of Chronos Studio."
    fix = "Pick one of the models listed in the Model registry panel."


class FitFailedError(ChronosError):
    """A model was offered but could not be estimated. Never fatal — always falls back."""

    code = "model.fit_failed"
    http_status = 422
    plain = "That model failed to fit on this data."
    fix = "Try a simpler order, a different family, or remove the exogenous variables."

    def __init__(self, model_id: str = "", reason: str = "", **kw: Any) -> None:
        super().__init__(context={"model_id": model_id, "reason": reason}, **kw)


class NotComparableError(ChronosError):
    """CRAFT §5.9: refuse AIC/BIC comparisons across different data transforms."""

    code = "model.not_comparable"
    http_status = 422
    plain = "These models cannot be compared with AIC/BIC because they were fitted to differently transformed data."
    fix = "Compare them with out-of-sample accuracy (rolling-origin CV) instead."


class LeakageError(ChronosError):
    code = "model.leakage"
    http_status = 500
    plain = "An internal check stopped the analysis because test-period data reached the model."
    fix = "This is a bug in Chronos Studio, not in your data. Please report it with the report archive."


class GateBlockedError(ChronosError):
    """The readiness gate (S8) has unresolved blockers."""

    code = "gate.blocked"
    http_status = 409
    plain = "The data is not ready for this step yet."
    fix = "Open the Readiness gate panel and clear the listed blockers first."


# --------------------------------------------------------------------------- #
# AI assistant
# --------------------------------------------------------------------------- #
class AIUnavailableError(ChronosError):
    code = "ai.unavailable"
    http_status = 503
    plain = "The AI assistant is not reachable right now."
    fix = "Keep working — the analysis and reports do not need the AI. Or check your Gemini API key in Settings."


class AIKeyMissingError(ChronosError):
    code = "ai.key_missing"
    http_status = 401
    plain = "No Gemini API key has been entered."
    fix = "Paste your free Google AI Studio key in Settings, or switch the assistant to Offline mode."


class AIRateLimitedError(ChronosError):
    code = "ai.rate_limited"
    http_status = 429
    plain = "The free AI quota has been used up for now."
    fix = "Wait a minute and try again, or use Offline mode — it needs no key and no quota."


class AIGroundingError(ChronosError):
    """The assistant produced a number that is not in the computed results."""

    code = "ai.ungrounded"
    http_status = 502
    plain = "The assistant's answer was withheld because it quoted a number the app did not compute."
    fix = "Ask again, or click 'Explain this' on the specific result card you are looking at."


class AIKeyInvalidError(ChronosError):
    code = "ai.key_invalid"
    http_status = 422
    plain = "That Gemini key looks malformed."
    fix = "Paste the key again from Google AI Studio, or choose Offline mode — no key is needed."


class AIPayloadTooLargeError(ChronosError):
    code = "ai.payload_too_large"
    http_status = 413
    plain = "That plot image is too large for the assistant."
    fix = "Attach a smaller PNG, JPEG or WebP screenshot, or use the stored figure-data payload instead."


class AIRequestCapError(ChronosError):
    code = "ai.request_cap"
    http_status = 429
    plain = "This analysis has reached its hard limit for assistant requests."
    fix = "Start a new analysis, or keep working with the computed results and offline explanations."


class AIPreviewRequiredError(ChronosError):
    code = "ai.preview_required"
    http_status = 428
    plain = "Review the assistant payload before the first live request."
    fix = "Open Payload preview, read what would be sent, then confirm it. Offline mode needs no review."


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class SessionNotFoundError(ChronosError):
    code = "session.not_found"
    http_status = 404
    plain = "That analysis session has expired."
    fix = "Upload the data again and replay the analysis; a .chronos file is a report export, not a resumable session."


class StageOrderError(ChronosError):
    code = "session.stage_order"
    http_status = 409
    plain = "That step needs an earlier step to be finished first."
    fix = "Follow the pipeline stepper from left to right."


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def as_error_payload(exc: BaseException) -> dict[str, Any]:
    """Serialize any exception for the API, without leaking a stack trace by default."""
    if isinstance(exc, ChronosError):
        return exc.to_dict()
    return {
        "code": "internal.error",
        "plain": "Something unexpected went wrong.",
        "fix": "Your data is safe. Reload the page and try again.",
        "technical": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:],
        "context": {},
    }


__all__ = [
    "ChronosError",
    "UnsupportedFormatError",
    "FileTooLargeError",
    "EmptyDatasetError",
    "TimeIndexError",
    "FrequencyInferenceError",
    "IrregularSpacingError",
    "ColumnMappingError",
    "NonNumericSeriesError",
    "NonPositiveDataError",
    "InsufficientDataError",
    "ModelNotFoundError",
    "FitFailedError",
    "NotComparableError",
    "LeakageError",
    "GateBlockedError",
    "AIUnavailableError",
    "AIKeyMissingError",
    "AIRateLimitedError",
    "AIGroundingError",
    "AIKeyInvalidError",
    "AIPayloadTooLargeError",
    "AIRequestCapError",
    "AIPreviewRequiredError",
    "SessionNotFoundError",
    "StageOrderError",
    "as_error_payload",
]
