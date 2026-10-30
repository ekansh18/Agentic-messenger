"""Function tools shared across the examples."""

from agents import function_tool

from .messenger import push, push_configured, send_message


@function_tool
def push_tool(message: str) -> str:
    """Send the given message to the user as a push notification."""
    if push_configured():
        push(message)
        return "Push sent"
    print(f"[console] Push: {message}")
    return "Pushover not configured; message printed to console"


@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """
    Send an email with the given subject and body to the user.

    Args:
        subject: The subject of the email
        text_body: The body of the email as plain text
        html_body: The HTML body of the email
    """
    channel = send_message(subject, text_body, html_body)
    return f"Message delivered via {channel}"
