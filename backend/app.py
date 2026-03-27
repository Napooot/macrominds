"""
backend/app.py

MacroMinds Flask application entry point.

Registers the API blueprint and starts the development server.

Usage
-----
    python -m backend.app          # from project root
    python backend/app.py          # direct
    flask --app backend.app run    # via Flask CLI
"""

import os
import sys
import logging

from flask import Flask

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.routes.api import api_bp  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(api_bp)

    @app.route('/health')
    def health():
        from flask import jsonify
        return jsonify({"status": "ok"})

    log.info("Registered routes:")
    with app.app_context():
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            log.info(f"  {rule.methods - {'HEAD', 'OPTIONS'}}  {rule.rule}")

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'true').lower() == 'true',
    )
