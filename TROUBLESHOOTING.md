# 🔧 Troubleshooting Deployment Issues

## Common Errors and Solutions

### Error: "Error installing requirements"

#### Solution 1: Check Streamlit Cloud Terminal
1. Streamlit Cloud dashboard me jao
2. "Manage App" click karo
3. "Terminal" tab me jao
4. Actual error message dekho

#### Solution 2: PyTorch Installation Issue
Agar PyTorch install nahi ho raha, to `requirements.txt` me yeh try karo:

```txt
# Remove torch line and let sentence-transformers handle it
# torch>=2.0.0,<3.0.0  # Comment this out
```

Ya minimal version:
```txt
torch==2.0.1
```

#### Solution 3: Version Conflicts
Agar specific package error aaye, to pinned versions use karo:

```txt
streamlit==1.39.0
pandas==2.2.0
numpy==1.26.0
spacy==3.7.2
sentence-transformers==2.7.0
transformers==4.44.2
torch==2.0.1
scikit-learn==1.5.0
pdfplumber==0.11.0
PyPDF2==3.0.1
python-docx==1.1.0
altair==5.2.0
pydantic==2.8.0
```

#### Solution 4: Memory Issues
Agar "Out of memory" error aaye:
- Streamlit Cloud free tier: 1GB RAM
- Try smaller models or upgrade tier

#### Solution 5: spaCy Model Download
Agar spaCy model error aaye, app automatically download karega. But agar issue ho, to:

```python
# app.py me add karo (already hai, but verify karo)
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
```

### Error: "Module not found"

**Solution:** Check `requirements.txt` me sab dependencies hain:
- ✅ streamlit
- ✅ pandas
- ✅ numpy
- ✅ spacy
- ✅ sentence-transformers
- ✅ transformers
- ✅ torch
- ✅ scikit-learn
- ✅ pdfplumber
- ✅ PyPDF2
- ✅ python-docx
- ✅ altair
- ✅ pydantic

### Error: "Build timeout"

**Solution:** 
- First deployment me models download hote hain (5-10 min)
- Wait karo, ya retry karo

### Error: "App crashed"

**Solution:**
1. Check terminal logs
2. Verify all imports work
3. Check if spaCy model downloaded properly

## Quick Fix Checklist

- [ ] Requirements.txt me sab packages hain
- [ ] Version conflicts nahi hain
- [ ] PyTorch compatible version hai
- [ ] Streamlit Cloud terminal me actual error check kiya
- [ ] App logs check kiye
- [ ] Retry kiya after fixing

## Alternative: Minimal Requirements

Agar still issues aaye, to yeh minimal version try karo:

```txt
streamlit
pandas
numpy
spacy
sentence-transformers
transformers
torch
scikit-learn
pdfplumber
PyPDF2
python-docx
altair
pydantic
```

(Without version pins - latest compatible versions install hongi)

## Still Having Issues?

1. **Streamlit Cloud Terminal** me exact error message share karo
2. **Logs** check karo
3. **GitHub Issues** me post karo with error details

---

**Most Common Fix:** PyTorch version issue - try `torch==2.0.1` ya remove torch line entirely.

