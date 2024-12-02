import requests
from bs4 import BeautifulSoup
import time
import os
from llama_cpp import Llama
import matplotlib.pyplot as plt
from urllib.parse import urljoin

class CommentScraper:
    def __init__(self, model_path, max_comments=35, retries=3):
        # Initialize the Llama model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
        )
        self.max_comments = max_comments
        self.retries = retries

    # Scrape comments from a URL with retry logic
    def scrape_comments(self, url):
        comments = []
        retries_left = self.retries  # Keep track of retries for each request
        while url and len(comments) < self.max_comments:
            try:
                print(f"Scraping {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise an exception for HTTP errors
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract comments (Adjust class selector as needed)
                reviews = soup.select('.fdbk-container__details__top')
                for review in reviews:
                    comment_text = review.get_text(strip=True)
                    if comment_text and len(comments) < self.max_comments:
                        comments.append(comment_text)

                # Check for next page link and construct full URL if necessary
                next_page = soup.select_one('.pagination__items a')
                if next_page:
                    next_page_url = next_page['href']
                    # Ensure next_page URL is absolute
                    url = urljoin(url, next_page_url)
                    print(f"Found next page: {url}")
                else:
                    print("No next page found.")
                    url = None

                # Introduce a small delay between requests to avoid overwhelming the server
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")
                retries_left -= 1
                if retries_left > 0:
                    print(f"Retrying {url} ({retries_left} retries left)...")
                    time.sleep(5)  # Wait before retrying
                else:
                    print(f"Failed after {self.retries} retries.")
                    break

        print(f"Collected {len(comments)} comments.")
        return comments

    # Get sentiment for a comment using Llama model
    def get_sentiment(self, comment):
        try:
            prompt = f"Evaluate the sentiment of this comment, respond with only positive, negative, or neutral: '{comment}'"
            response = self.llm(
                prompt,
                max_tokens=10,  # We only need a short response for sentiment
                stop=["<|end|>"],
                echo=False,
            )

            sentiment = response['choices'][0]['text'].strip().lower()

            # Debugging output to inspect the raw model response
            print(prompt + '\n')
            print(sentiment)

            # Refined sentiment detection
            if "positive" in sentiment:
                return "positive"
            elif "negative" in sentiment:
                return "negative"
            else:
                return "neutral"

        except Exception as e:
            print(f"Error with Llama sentiment analysis: {e}")
            return "neutral"

    # Save comments and their sentiments to 'comments.txt'
    def save_comments_with_sentiment(self, comments, sentiments):
        with open('comments.txt', 'a', encoding='utf-8') as file:  # Open in append mode
            for comment, sentiment in zip(comments, sentiments):
                file.write(f"Comment: {comment}\nSentiment: {sentiment}\n\n")
        print(f"Appended {len(comments)} comments and sentiments to 'comments.txt'.")

    # Read URLs from the file
    @staticmethod
    def read_urls_from_file(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return []

    # Function to create and display a sentiment bar graph for each URL
    def create_sentiment_graph(self, url, sentiments):
        # Count the occurrences of each sentiment type
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for sentiment in sentiments:
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Plot the bar graph
        labels = sentiment_counts.keys()
        values = sentiment_counts.values()

        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, color=['green', 'red', 'gray'])
        plt.title(f"Sentiment Analysis Graph")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Comments")
        plt.tight_layout()

        # Save the graph to the current directory (no folder)
        graph_filename = "sentiment_graph.png"
        plt.savefig(graph_filename)
        print(f"Saved sentiment graph as {graph_filename}")
        plt.close()

    # Main function to execute the process for all URLs
    def process_comments(self, urls):
        for url in urls:
            print(f"Processing URL: {url}...")
            comments = self.scrape_comments(url)
            if comments:
                # Get sentiments for each comment
                sentiments = [self.get_sentiment(comment) for comment in comments]
                
                # Append comments and sentiments to 'comments.txt'
                self.save_comments_with_sentiment(comments, sentiments)
                
                # Create and save the sentiment graph for the URL
                self.create_sentiment_graph(url, sentiments)
            else:
                print(f"No comments found for {url}.")

# Example of how to use the class:
if __name__ == "__main__":
    # Optionally, remove 'comments.txt' if it exists (start fresh)
    if os.path.exists('comments.txt'):
        os.remove('comments.txt')

    # Initialize the scraper with model path
    scraper = CommentScraper(
        model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # Adjust model path if necessary
    )

    # Read URLs from a file and start the process
    urls = scraper.read_urls_from_file('Links.txt')
    if urls:
        scraper.process_comments(urls)
