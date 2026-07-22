import os
import re
import urllib.parse
import requests
import pandas as pd
from groq import Groq
from duckduckgo_search import DDGS

# Hardcoded Groq API Key
GROQ_API_KEY = ""

# Available Models (Prioritize llama-3.3-70b-versatile, fallback to llama-3.1-8b-instant, etc.)
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama3-70b-8192"]

class DatasetSearchAgent:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODELS[0]
        
    def log(self, message):
        try:
            print(message)
        except UnicodeEncodeError:
            try:
                print(message.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass
        if self.log_callback:
            self.log_callback(message)

    def call_llm(self, system_prompt, user_prompt):
        """Helper to invoke Groq API with robust model fallback."""
        for model in MODELS:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model,
                    temperature=0.2,
                    max_tokens=1000
                )
                self.model = model  # Keep track of active model
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                self.log(f"⚠️ Warning: Failed with model {model}: {str(e)}. Trying next model...")
        raise Exception("Failed to query Groq API with any available model.")

    def optimize_search_query(self, user_query):
        """Asks the LLM to write an optimized search query to find direct CSV downloads."""
        self.log(f"🧠 Agent is thinking: Optimizing query for '{user_query}'...")
        system_prompt = (
            "You are an expert data sourcing assistant. Your job is to convert a user's request for a dataset "
            "into a simple search query targeting raw CSV files. "
            "Do NOT use advanced search operators like filetype:csv, OR, AND, site: etc., as DuckDuckGo's HTML search does not support them well. "
            "Stick to plain text keywords. Example output: 'iris dataset raw csv github' or 'heart disease dataset raw csv repository'."
            "Respond ONLY with the search query. Do not include quotes, explanations, or any extra text."
        )
        user_prompt = (
            f"User wants: '{user_query}'. "
            f"Format a search query that will find raw direct download links or repositories containing this CSV."
        )
        optimized = self.call_llm(system_prompt, user_prompt)
        # Clean any accidental quotes
        optimized = re.sub(r'^["\']|["\']$', '', optimized)
        self.log(f"🔍 Optimized search query: '{optimized}'")
        return optimized

    def search_web(self, query):
        """Searches DuckDuckGo for matching web pages."""
        self.log(f"🌐 Searching the web using DuckDuckGo...")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))
            if not results:
                self.log("⚠️ No results returned from DuckDuckGo wrapper. Trying fallback search...")
                return self.fallback_search(query)
            self.log(f"Found {len(results)} search results.")
            return results
        except Exception as e:
            self.log(f"❌ Error during web search wrapper: {str(e)}")
            return self.fallback_search(query)

    def fallback_search(self, query):
        """Fallback search using simple requests scraping to ensure high reliability."""
        self.log("🔄 Using fallback web search scraper...")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                self.log(f"❌ Fallback search failed with status {r.status_code}")
                return []
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')
            results = []
            for result_div in soup.find_all('div', class_='result'):
                try:
                    title_elem = result_div.find('a', class_='result__snippet')
                    if not title_elem:
                        title_elem = result_div.find('a', class_='result__url')
                    if not title_elem:
                        continue
                    
                    href_elem = result_div.find('a', class_='result__url')
                    if not href_elem:
                        continue
                    
                    href = href_elem.get('href', '')
                    title = href_elem.text.strip()
                    
                    snippet_elem = result_div.find('a', class_='result__snippet') or result_div.find('span', class_='result__snippet')
                    snippet = snippet_elem.text.strip() if snippet_elem else ""
                    
                    # Resolve DDG redirect link
                    parsed_url = urllib.parse.urlparse(href)
                    qs = urllib.parse.parse_qs(parsed_url.query)
                    actual_url = qs.get('uddg', [href])[0]
                    
                    results.append({
                        "title": title,
                        "href": actual_url,
                        "body": snippet
                    })
                except Exception:
                    continue
            self.log(f"Fallback search found {len(results)} results.")
            return results[:10]
        except Exception as e:
            self.log(f"❌ Fallback search failed: {str(e)}")
            return []

    def analyze_results(self, search_results, original_query):
        """Asks the LLM to rank and select the best URLs to download."""
        self.log("🧠 Agent is analyzing search results to find the best CSV link...")
        
        results_formatted = ""
        for idx, r in enumerate(search_results):
            results_formatted += f"[{idx}] Title: {r.get('title')}\nURL: {r.get('href')}\nDescription: {r.get('body')}\n\n"

        system_prompt = (
            "You are an AI assistant that finds raw CSV dataset files. "
            "You are given a list of search results. Your goal is to return a ranked list of indices (from best to worst) "
            "pointing directly to raw CSV files or download pages. "
            "Prioritize URLs that end in '.csv', or are from GitHub raw, kaggle, archive.ics.uci.edu, raw.githubusercontent.com, etc. "
            "Respond ONLY with a comma-separated list of indices (e.g., '3,0,5'). No other text."
        )
        user_prompt = (
            f"Original Query: '{original_query}'\n\n"
            f"Search Results:\n{results_formatted}\n"
            f"Return the best candidate indices as a comma-separated list."
        )
        
        response = self.call_llm(system_prompt, user_prompt)
        indices = []
        for val in response.split(','):
            try:
                idx = int(val.strip())
                if 0 <= idx < len(search_results):
                    indices.append(idx)
            except ValueError:
                continue
        
        candidates = [search_results[i] for i in indices]
        # Append others that weren't selected just in case
        for idx, r in enumerate(search_results):
            if r not in candidates:
                candidates.append(r)
                
        return candidates

    def download_and_verify(self, candidate_url, filename):
        """Downloads the URL, verifies if it's a valid CSV, and saves it if correct."""
        self.log(f"📥 Attempting download from: {candidate_url}")
        
        # Clean URL if it's GitHub to point to raw content
        if "github.com" in candidate_url and "/blob/" in candidate_url:
            raw_url = candidate_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            self.log(f"🔄 Converting GitHub blob URL to Raw URL: {raw_url}")
        else:
            raw_url = candidate_url

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            response = requests.get(raw_url, headers=headers, timeout=15, stream=True)
            
            if response.status_code != 200:
                self.log(f"❌ HTTP Error {response.status_code} for {raw_url}")
                return False, None
            
            # Check content-type or size
            content_type = response.headers.get('Content-Type', '').lower()
            if 'html' in content_type:
                # Often it's a web page, not a raw file
                self.log("⚠️ Content-type is HTML, might not be a raw CSV file. Analyzing preview...")
            
            # Read first few KB to verify CSV structure
            chunk = response.raw.read(1024 * 100) # 100 KB
            text = chunk.decode('utf-8', errors='ignore')
            
            # Use Groq to quickly verify if the text content looks like a CSV (fallback to pandas)
            is_valid_csv = False
            try:
                # Save temp to verify with Pandas
                temp_filename = "temp_check.csv"
                with open(temp_filename, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Check with pandas (can read first few lines)
                df = pd.read_csv(temp_filename, nrows=5)
                if len(df.columns) > 1 or (len(df.columns) == 1 and ',' in text):
                    is_valid_csv = True
                os.remove(temp_filename)
            except Exception:
                if os.path.exists("temp_check.csv"):
                    os.remove("temp_check.csv")
                # LLM check as fallback
                self.log("🔄 Validating file structure using LLM parser...")
                system_prompt = "You are a CSV verification tool. Output ONLY 'VALID' or 'INVALID' based on whether the input text looks like a standard CSV file (header row followed by comma or semicolon separated values)."
                res = self.call_llm(system_prompt, text[:2000])
                if "VALID" in res.upper():
                    is_valid_csv = True

            if not is_valid_csv:
                self.log("❌ File does not appear to be a valid CSV dataset.")
                return False, None

            # Complete the download
            filepath = os.path.join(os.getcwd(), filename)
            # Write full content
            # Re-read full response content since we read a chunk
            full_response = requests.get(raw_url, headers=headers, timeout=15)
            with open(filepath, "wb") as f:
                f.write(full_response.content)
            
            self.log(f"✅ Success! Dataset saved to: {filepath}")
            
            # Load basic info for preview
            df_preview = pd.read_csv(filepath)
            info = {
                "filepath": filepath,
                "shape": df_preview.shape,
                "columns": list(df_preview.columns),
                "head": df_preview.head(5).to_dict(orient='records'),
                "summary": df_preview.describe(include='all').to_dict()
            }
            return True, info
            
        except Exception as e:
            self.log(f"❌ Error during download/verification: {str(e)}")
            return False, None

    def execute_agent(self, user_query, target_filename):
        """Runs the complete agent flow."""
        self.log(f"🚀 Starting AI Agent for dataset: '{user_query}'")
        
        # 1. Optimize search query
        query = self.optimize_search_query(user_query)
        
        # 2. Search web
        results = self.search_web(query)
        if not results:
            self.log("❌ Sourcing failed: No web results found.")
            return None
            
        # 3. Analyze and select candidates
        candidates = self.analyze_results(results, user_query)
        
        # 4. Try downloading and verifying candidates
        for idx, candidate in enumerate(candidates):
            self.log(f"🔍 Sourcing Candidate {idx+1}/{len(candidates)}: {candidate.get('title')}")
            success, info = self.download_and_verify(candidate.get('href'), target_filename)
            if success:
                self.log("🎉 Agent completed the task successfully!")
                return info
            self.log("🔄 Trying next candidate URL...")
            
        self.log("❌ Sourcing failed: Checked all candidates but none resulted in a valid CSV.")
        return None
