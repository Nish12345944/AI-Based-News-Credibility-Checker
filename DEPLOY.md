# Deployment Guide

## Heroku Deployment

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

## Railway Deployment

1. Connect GitHub repo to Railway
2. Deploy automatically

## Render Deployment

1. Connect GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `cd backend && python app.py`

## Local Production

```bash
pip install -r requirements.txt
cd backend
python app.py
```

Access at: `http://localhost:8000`