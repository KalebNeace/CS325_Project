import pytest
from unittest.mock import patch, MagicMock
from FinalProject import CommentScraper 


# Test case 1: Test the sentiment analysis method
def test_get_sentiment_positive():
    scraper = CommentScraper(model_path="./Phi-3-mini-4k-instruct-q4.gguf")
    comment = "I love this product! It's amazing."
    sentiment = scraper.get_sentiment(comment)
    assert sentiment == "positive", f"Expected 'positive', but got {sentiment}"

def test_get_sentiment_negative():
    scraper = CommentScraper(model_path="./Phi-3-mini-4k-instruct-q4.gguf")
    comment = "This product is terrible. I hate it."
    sentiment = scraper.get_sentiment(comment)
    assert sentiment == "negative", f"Expected 'negative', but got {sentiment}"

def test_get_sentiment_neutral():
    scraper = CommentScraper(model_path="./Phi-3-mini-4k-instruct-q4.gguf")
    comment = "The product is okay, nothing special."
    sentiment = scraper.get_sentiment(comment)
    assert sentiment == "neutral", f"Expected 'neutral', but got {sentiment}"

# Test case 2: Test if URL mapping to filenames works correctly
def test_map_url_to_filename():
    scraper = CommentScraper(model_path="./Phi-3-mini-4k-instruct-q4.gguf")
    
    url = "https://example.com/product1/reviews"
    filename = scraper.map_url_to_filename(url, 0)
    assert filename == "galaxys22_comments.txt", f"Expected 'galaxys22_comments.txt', but got {filename}"
    
    url2 = "https://example.com/product2/reviews"
    filename2 = scraper.map_url_to_filename(url2, 1)
    assert filename2 == "galaxys21_comments.txt", f"Expected 'galaxys21_comments.txt', but got {filename2}"

# Test case 3: Test if the scraping function handles the comments correctly (mocking HTTP request)
@patch("requests.get")
def test_scrape_comments(mock_get):
    # Create a fake HTML response to mock the 'requests.get' call
    mock_response = MagicMock()
    mock_response.text = """
        <html>
            <div class="fdbk-container__details__top">Great product!</div>
            <div class="fdbk-container__details__top">Not bad, could be better.</div>
            <a href="page2.html" class="pagination__items">Next</a>
        </html>
    """
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    scraper = CommentScraper(model_path="./Phi-3-mini-4k-instruct-q4.gguf")
    url = "https://example.com/product/reviews"
    
    # Test scraping logic with a mocked request
    comments = scraper.scrape_comments(url)
    
    assert len(comments) == 2, f"Expected 2 comments, but got {len(comments)}"
    assert comments == ["Great product!", "Not bad, could be better."], f"Unexpected comments: {comments}"

# Test case 4: Test saving comments with sentiments to a file
def test_save_comments_with_sentiment(tmp_path):
    scraper = CommentScraper(model_path="./Phi-3-mini-4k-instruct-q4.gguf")
    comments = ["Great product!", "Not bad, could be better."]
    sentiments = ["positive", "neutral"]
    
    # Create a temporary file using pytest's tmp_path fixture
    test_file = tmp_path / "test_comments.txt"
    scraper.save_comments_with_sentiment(test_file, comments, sentiments)
    
    # Check if the file was created and has the correct content
    assert test_file.exists(), f"File {test_file} was not created"
    
    # Read the content of the file and check its correctness
    with open(test_file, "r", encoding="utf-8") as file:
        content = file.read()
    
    expected_content = "Comment: Great product!\nSentiment: positive\n\n"
    expected_content += "Comment: Not bad, could be better.\nSentiment: neutral\n\n"
    assert content == expected_content, f"Expected content: {expected_content}, but got: {content}"

