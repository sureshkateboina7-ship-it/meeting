# Client Package Manifest

Included:

- Application source code
- Windows setup and run scripts
- Client setup README
- Pinned Python requirements
- Local YOLO model weights in `models/`
- Database schema
- Static CSS/JS and templates

Excluded from the client zip:

- Local `venv/`
- Local `.env`
- Runtime logs
- Existing `database/violations.db`
- Existing uploaded and processed image/video files
- Python cache folders

The setup script recreates the venv, initializes a fresh database, verifies bundled YOLO model files, and downloads PaddleOCR runtime model assets into the client user's cache.
