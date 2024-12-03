# Kaleb Neace
# This program reads in links from Links.txt, then it navigates and pull the comments from those links. Next, it passes the comments into Phi-3
# and analyzes the sentiment of the comments. The comments and sentiments are saved in a text file specific to it's link.
# Lastly it creates a graph to visualizes the data from each link.

import requests
from bs4 import BeautifulSoup
import time
import os
from llama_cpp import Llama
import matplotlib.pyplot as plt
from urllib.parse import urljoin
import numpy as np

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

                # Extract comments
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

    # Save comments and their sentiments to a separate file for each URL
    def save_comments_with_sentiment(self, comments, sentiments, filename):
        with open(filename, 'w', encoding='utf-8') as file: 
            for comment, sentiment in zip(comments, sentiments):
                file.write(f"Comment: {comment}\nSentiment: {sentiment}\n\n")
        print(f"Saved comments and sentiments to '{filename}'.")

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
    def create_sentiment_graph(self, urls, all_sentiments, url_to_filename_mapping):
        # Prepare the data for plotting
        sentiment_counts_per_url = []

        # Calculate sentiment counts for each URL
        for sentiments in all_sentiments:
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for sentiment in sentiments:
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
            sentiment_counts_per_url.append(sentiment_counts)

        # Create an index for each group of bars (representing each URL)
        num_urls = len(urls)
        bar_width = 0.2  # Width of each bar in the group
        index = np.arange(num_urls)  # X-axis positions for each URL group

        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create the bars for each sentiment (positive, negative, neutral)
        positive_counts = [counts['positive'] for counts in sentiment_counts_per_url]
        negative_counts = [counts['negative'] for counts in sentiment_counts_per_url]
        neutral_counts = [counts['neutral'] for counts in sentiment_counts_per_url]

        # Plot bars for each sentiment type
        ax.bar(index - bar_width, positive_counts, bar_width, label='Positive', color='green')
        ax.bar(index, negative_counts, bar_width, label='Negative', color='red')
        ax.bar(index + bar_width, neutral_counts, bar_width, label='Neutral', color='gray')

        # Use the URL-to-filename mapping for titles
        mapped_titles = [url_to_filename_mapping.get(url, url) for url in urls]

        # Add labels, title, and legend
        ax.set_xlabel('URLs')
        ax.set_ylabel('Number of Comments')
        ax.set_title('Sentiment Distribution for Multiple URLs')
        ax.set_xticks(index)
        ax.set_xticklabels(mapped_titles, rotation=45, ha='right')
        ax.legend()

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the graph as a single PNG image
        graph_filename = "combined_sentiment_graph.png"
        plt.savefig(graph_filename)
        print(f"Saved combined sentiment graph as {graph_filename}")
        plt.close()

    # Main function to execute the process for all URLs
    def process_comments(self, urls, url_to_filename_mapping):
        all_sentiments = []
        
        for url in urls:
            print(f"Processing URL: {url}...")
            comments = self.scrape_comments(url)
            if comments:
                # Get sentiments for each comment
                sentiments = [self.get_sentiment(comment) for comment in comments]
                
                # Get the corresponding filename for this URL
                filename = url_to_filename_mapping.get(url, "default_comments.txt")
                
                # Save the comments and sentiments to a separate file
                self.save_comments_with_sentiment(comments, sentiments, filename)
                
                # Append sentiments for this URL to the all_sentiments list
                all_sentiments.append(sentiments)
            else:
                print(f"No comments found for {url}.")
                all_sentiments.append([])  # Add an empty list if no comments

        # After processing all URLs, create the combined graph
        self.create_sentiment_graph(urls, all_sentiments, url_to_filename_mapping)


if __name__ == "__main__":
    # Remove existing files for a fresh start
    filenames = ["galaxys22_comments.txt", "galaxys21_comments.txt", "galaxys20_comments.txt", "galaxys10_comments.txt"]
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)

    # Initialize the scraper with model path
    scraper = CommentScraper(
        model_path="./Phi-3-mini-4k-instruct-q4.gguf"
    )

    # Read URLs from the file and start the process
    urls = scraper.read_urls_from_file('Links.txt')
    
    # Define a mapping of URLs to their corresponding file names
    url_to_filename_mapping = {
        "https://www.ebay.com/fdbk/mweb_profile?fdbkType=FeedbackReceivedAsSeller&item_id=125792100860&username=everythingforlesss&filter=feedback_page%3ARECEIVED_AS_SELLER&q=125792100860&sort=RELEVANCE&page_id_item=1&sort_item=TIME&filter_image_item=false": "galaxys22_comments.txt",
        "https://www.ebay.com/fdbk/mweb_profile?fdbkType=FeedbackReceivedAsSeller&item_id=125641047454&username=everythingforlesss&filter=feedback_page%3ARECEIVED_AS_SELLER&q=125641047454&sort=RELEVANCE&page_id_item=1&sort_item=TIME&filter_image_item=false": "galaxys21_comments.txt",
        "https://www.ebay.com/fdbk/mweb_profile?fdbkType=FeedbackReceivedAsSeller&item_id=116088694961&username=everythingforlesss&filter=feedback_page%3ARECEIVED_AS_SELLER&q=116088694961&sort=RELEVANCE": "galaxys20_comments.txt",
        "https://www.ebay.com/fdbk/mweb_profile?fdbkType=FeedbackReceivedAsSeller&item_id=254497146992&username=cellfeee&filter=feedback_page%3ARECEIVED_AS_SELLER&q=254497146992&sort=RELEVANCE&page_id_item=1&sort_item=TIME&filter_image_item=false": "galaxys10_comments.txt",
    }
    
    if urls:
        scraper.process_comments(urls, url_to_filename_mapping)

