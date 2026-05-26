"""
Flask application factory for the flame3d-core server.
"""

from flask import Flask

from server.config import config_by_name


def create_app(config_name=None):
    """Create and configure the Flask application.

    Args:
        config_name: Configuration name ('development', 'production', 'testing').
                     Defaults to the FLASK_ENV environment variable, or 'development'.
    """
    import os

    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, config_by_name["development"]))

    # Register blueprints
    from server.routes import register_blueprints
    register_blueprints(app)

    return app
