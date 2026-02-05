"""Load and parse email files from disk."""

import logging
import re
from pathlib import Path

from rag.config import EMAILS_DIR
from rag.models import ParsedEmail

logger = logging.getLogger(__name__)

# Pattern: "From: Name <email>" or "To: Name <email>"
FROM_TO_PATTERN = re.compile(r"^(From|To):\s*(.+?)\s*<(.+?)>\s*$", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"^Subject:\s*(.+)$", re.IGNORECASE)


def parse_email_content(content: str, source_file: str) -> ParsedEmail | None:
    """
    Parse raw email text into structured fields.
    Returns None if parsing fails (invalid format).
    """
    lines = content.splitlines()
    subject = ""
    from_name = ""
    from_email = ""
    to_name = ""
    to_email = ""
    body_lines: list[str] = []
    header_done = False

    for line in lines:
        if not header_done:
            sub_m = SUBJECT_PATTERN.match(line.strip())
            if sub_m:
                subject = sub_m.group(1).strip()
                continue
            ft_m = FROM_TO_PATTERN.match(line.strip())
            if ft_m:
                label, name, email = ft_m.group(1), ft_m.group(2).strip(), ft_m.group(3).strip()
                if label.lower() == "from":
                    from_name, from_email = name, email
                else:
                    to_name, to_email = name, email
                continue
            # Empty line after we have at least To (and usually From) ends header
            if line.strip() == "" and (from_email or to_email):
                header_done = True
                continue
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if not subject and not from_email and not to_email:
        logger.warning("Could not parse email %s: missing headers", source_file)
        return None
    return ParsedEmail(
        source_file=source_file,
        subject=subject,
        from_name=from_name,
        from_email=from_email,
        to_name=to_name,
        to_email=to_email,
        body=body,
    )


def load_email_file(path: Path) -> ParsedEmail | None:
    """Load a single email file and return ParsedEmail or None."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return parse_email_content(content, path.name)
    except OSError as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def load_all_emails(emails_dir: Path | None = None) -> list[ParsedEmail]:
    """
    Load all email_*.txt files from the given directory.
    Returns list of successfully parsed emails; logs and skips failures.
    """
    directory = emails_dir or EMAILS_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Emails directory not found: {directory}")

    emails: list[ParsedEmail] = []
    paths = sorted(directory.glob("email_*.txt"))
    for path in paths:
        parsed = load_email_file(path)
        if parsed:
            emails.append(parsed)
        else:
            logger.warning("Skipped unparseable file: %s", path.name)

    logger.info("Loaded %d emails from %s", len(emails), directory)
    return emails
