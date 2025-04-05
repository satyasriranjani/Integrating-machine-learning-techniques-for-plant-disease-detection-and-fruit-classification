from flask import Flask

def create_app(config_filename='config.py'):
    # app = Flask(__name__, instance_relative_config=True)
    # app.config.from_pyfile(config_filename)

    # Register blueprints
    from app.routes import app
    # app.register_blueprint(app)

    return app