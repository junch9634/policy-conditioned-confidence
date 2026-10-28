from typing import Literal, Optional
Verdict = Literal["yes", "no", "unknown"]


def parse_event_verdict(text: Optional[str]) -> Optional[Verdict]:
    """
    첫 번째 판정어 추출. 형식: '네, ...' / '아니요, ...' / '판단불가...'
    영문 Yes/No/Unknown 도 허용.
    """
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    head = s.split(",")[0].strip()
    if head == "판" or head.startswith("판단불가"):
        return "unknown"
    if head == "아" or head.startswith("아니요"):
        return "no"
    if head == "네" or head.startswith("네"):
        return "yes"
    hl = head.lower()
    if hl.startswith("yes"):
        return "yes"
    if hl.startswith("no"):
        return "no"
    if hl.startswith("unknown"):
        return "unknown"
    return None


