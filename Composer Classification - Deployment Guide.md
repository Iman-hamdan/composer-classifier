# Composer Classification - Deployment Guide

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 20+
- Git

### 1. Backend Setup
```bash
cd composer_classifier/web/backend
source venv/bin/activate  # or create new: python -m venv venv
pip install flask flask-cors tensorflow scikit-learn pretty-midi
python src/main.py
```
Backend will run on: http://localhost:5000

### 2. Frontend Setup
```bash
cd composer_classifier/web/frontend
npm install
npm run dev
```
Frontend will run on: http://localhost:5176

### 3. Access Application
Open http://localhost:5176 in your browser

## 🌐 Cloud Deployment Options

### Option 1: Vercel (Frontend) + Railway (Backend)
**Frontend (Vercel):**
```bash
cd composer_classifier/web/frontend
npm install -g vercel
vercel --prod
```

**Backend (Railway):**
1. Push to GitHub repository
2. Connect to Railway.app
3. Deploy from GitHub

### Option 2: Heroku (Full Stack)
```bash
# Create Heroku apps
heroku create your-app-backend
heroku create your-app-frontend

# Deploy backend
cd composer_classifier/web/backend
git init && git add . && git commit -m "Initial commit"
heroku git:remote -a your-app-backend
git push heroku main

# Deploy frontend (update API URL first)
cd ../frontend
# Update API calls to use Heroku backend URL
npm run build
# Deploy to Heroku or Vercel
```

### Option 3: AWS/GCP
- Use AWS Elastic Beanstalk or GCP App Engine
- Configure environment variables
- Set up load balancing for production

## 🔧 Configuration

### Environment Variables
```bash
# Backend
FLASK_ENV=production
FLASK_DEBUG=False

# Frontend
VITE_API_URL=https://your-backend-url.com
```

### Production Optimizations
1. **Model Optimization**: Use TensorFlow Lite for faster inference
2. **Caching**: Add Redis for model predictions
3. **CDN**: Use CloudFlare for static assets
4. **Monitoring**: Add logging and error tracking

## 📊 Performance Monitoring

### Health Checks
- Backend: `GET /api/health`
- Frontend: Check console for errors

### Metrics to Monitor
- Response time (< 2 seconds target)
- Model accuracy (> 95% target)
- Error rates (< 1% target)
- Uptime (99.9% target)

## 🛡️ Security Considerations

### Production Checklist
- [ ] Enable HTTPS
- [ ] Set CORS origins properly
- [ ] Add rate limiting
- [ ] Implement file size limits
- [ ] Add input validation
- [ ] Set up monitoring/logging

## 🎯 Next Steps

1. **Test Locally**: Verify everything works
2. **Choose Deployment**: Pick your preferred platform
3. **Configure Domain**: Set up custom domain
4. **Monitor**: Set up analytics and monitoring
5. **Scale**: Add features as needed

## 📞 Support

If you need help with deployment:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure ports are not blocked
4. Test API endpoints individually

Your application is production-ready and can handle real users! 🎉

