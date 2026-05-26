"""
WSGI entry point for the Flask server.

Usage:
    python -m server.wsgi
    # or via Flask CLI:
    flask --app server:create_app() run
"""

from server import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
