# 🚀 GitHub Setup Instructions

## Step 1: GitHub Repository Create Karein

1. **GitHub.com pe jao** aur login karo
2. **New Repository** button click karo (top right me + icon)
3. Repository details:
   - **Repository name**: `ai-resume-screener` (ya apna naam)
   - **Description**: "AI-powered resume screening tool using NLP and embeddings"
   - **Visibility**: Public (ya Private - apki choice)
   - **Initialize**: ❌ README, .gitignore, license ko UNCHECK karo (hamare paas already hai)
4. **Create repository** click karo

## Step 2: Local Git Initialize Karein

Terminal/PowerShell me project folder me jao:

```bash
cd "C:\Users\dhruv singh\Downloads\AI RESUME"
```

### Git Initialize:

```bash
# Git initialize karo
git init

# Sab files ko stage karo
git add .

# First commit
git commit -m "Initial commit: AI Resume Screener with multi-factor scoring"
```

## Step 3: GitHub Repository Se Connect Karein

GitHub pe jo repository banaya, uska URL copy karo (jaise: `https://github.com/yourusername/ai-resume-screener.git`)

```bash
# Remote repository add karo (apna URL use karo)
git remote add origin https://github.com/YOUR_USERNAME/ai-resume-screener.git

# Branch name set karo (main ya master)
git branch -M main

# Push karo
git push -u origin main
```

**Note**: Agar pehli baar GitHub use kar rahe ho, to authentication ke liye:
- **Personal Access Token** chahiye hoga (Settings → Developer settings → Personal access tokens)
- Ya **GitHub CLI** install karo

## Step 4: Verification

GitHub repository page pe jao - sab files dikhni chahiye:
- ✅ app.py
- ✅ requirements.txt
- ✅ README.md
- ✅ src/ folder
- ✅ .gitignore

## 🔄 Future Updates Push Karne Ke Liye

Jab bhi changes karo:

```bash
# Changes check karo
git status

# Files add karo
git add .

# Commit karo
git commit -m "Description of changes"

# Push karo
git push
```

## 📝 Optional: GitHub Features

### 1. **Topics/Tags Add Karein** (Repository settings me):
- `python`
- `nlp`
- `resume-screening`
- `streamlit`
- `machine-learning`
- `recruitment`

### 2. **About Section** (Repository page pe):
- Short description: "AI-powered resume screening using NLP"
- Website: (agar deploy kiya ho)
- Topics: (upar wale tags)

### 3. **README Badges** (already added):
- Python version
- Streamlit version
- License

## 🐛 Common Issues

### Issue: "Authentication failed"
**Solution**: Personal Access Token use karo (password ki jagah)

### Issue: "Repository not found"
**Solution**: Repository URL check karo, username sahi hai?

### Issue: "Permission denied"
**Solution**: GitHub account me repository access hai ya nahi check karo

## ✅ Checklist

- [ ] GitHub repository create ho gaya
- [ ] Git initialized
- [ ] Files committed
- [ ] Remote added
- [ ] Code pushed
- [ ] README properly display ho raha hai
- [ ] .gitignore kaam kar raha hai (__pycache__ exclude ho raha hai)

---

**Happy Coding! 🎉**

