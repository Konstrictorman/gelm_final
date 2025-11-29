# API Key Setup Guide

## Quick Setup

You need to set your OpenAI API key before running the notebook. Here are three methods:

### Method 1: Set Directly in Notebook (Quick but less secure)

In the **"Setup API keys"** cell, uncomment and add your key:

```python
os.environ["OPENAI_API_KEY"] = "sk-your-actual-api-key-here"
```

⚠️ **Warning**: Don't commit this to version control! Remove it before pushing to GitHub.

### Method 2: Environment Variable (Recommended)

**In your terminal**, before starting Jupyter:

```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

Then start Jupyter from that same terminal:

```bash
jupyter notebook
```

### Method 3: .env File (Best for local development)

1. **Install python-dotenv**:

   ```bash
   pip install python-dotenv
   ```

2. **Create a `.env` file** in your project directory:

   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   TAVILY_API_KEY=your-tavily-key-here  # Optional
   ```

3. **Add `.env` to `.gitignore`** (already done if you have one)

4. The notebook will automatically load it!

## Getting Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in to your OpenAI account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)
5. **Save it somewhere safe** - you won't be able to see it again!

## Optional: Tavily API Key

Tavily is used for web search. If you don't have a Tavily key, the notebook will automatically use DuckDuckGo (free, no API key needed).

To get a Tavily key:

1. Go to https://tavily.com
2. Sign up for an account
3. Get your API key from the dashboard

## Verification

After setting up your API key, run the **"Setup API keys"** cell. You should see:

```
✅ OPENAI_API_KEY is set
```

If you see a warning, follow the instructions in the cell output.

## Troubleshooting

**Error: "OPENAI_API_KEY is not set"**

- Make sure you've set the key using one of the methods above
- If using environment variables, restart Jupyter after setting them
- If using .env file, make sure it's in the same directory as the notebook

**Error: "Invalid API key"**

- Check that your key starts with `sk-`
- Make sure you copied the entire key (no extra spaces)
- Verify the key is active in your OpenAI dashboard

## Security Best Practices

✅ **DO**:

- Use environment variables or .env files
- Add .env to .gitignore
- Use different keys for development and production
- Rotate keys periodically

❌ **DON'T**:

- Commit API keys to version control
- Share keys in screenshots or messages
- Use production keys in development notebooks
