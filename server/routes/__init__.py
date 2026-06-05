"""
Blueprint registration for the Flask server.
"""

from flask import Flask


def register_blueprints(app: Flask):
    """Register all blueprints with the Flask application."""
    from server.routes.health import health_bp
    from server.routes.processing import processing_bp
    from server.routes.search import search_bp
    from server.routes.component_view import component_view_bp
    from server.routes.component_edit import component_edit_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(processing_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(component_view_bp)
    app.register_blueprint(component_edit_bp)
