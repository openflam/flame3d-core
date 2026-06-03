"""
Blueprint registration for the Flask server.
"""

from flask import Flask


def register_blueprints(app: Flask):
    """Register all blueprints with the Flask application."""
    from server.routes.health import health_bp
    from server.routes.processing import processing_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(processing_bp)
