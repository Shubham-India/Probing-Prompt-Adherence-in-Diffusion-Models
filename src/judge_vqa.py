"""J2 interface: atomic-question judge on top of a pluggable vision-language backend (Soft-TIFA-style, RB 12.5).
NO VLM is selected or loaded here: the choice depends on E03 (VRAM fit on the Kaggle GPU) and E01 (accuracy on the DEV audit).
A backend implements ask(image, question) -> p_yes in [0, 1]. Constraints use the legacy format, so J1 and J2 are interchangeable
in the analysis. Items with p_yes in the uncertainty band are flagged, never silently rounded."""
from audit_sample import question

BAND = (0.35, 0.65)


class Backend:
    def ask(self, image, question_text):   # -> probability that the answer is "yes"
        raise NotImplementedError("plug in a VLM backend after E03")


def score_constraints(image, constraints, backend, band=BAND):
    """Returns (results, probs, uncertain): results are 0/1 by p_yes >= 0.5; uncertain=1 if any p_yes lies inside the band."""
    probs = [float(backend.ask(image, question(c))) for c in constraints]
    return [int(p >= 0.5) for p in probs], probs, int(any(band[0] <= p <= band[1] for p in probs))
