import pytest
from unittest import mock
from FinalProject import CommentScraper

# Test scraping comments functionality
@pytest.fixture
def mock_html_response():
    return """
    <html>
        <body>
            <div class="fdbk-container__details__top">Great product!</div>
            <div class="fdbk-container__details__top">Very satisfied with my purchase.</div>
            <div class="fdbk-container__details__top">Not as expected.</div>
        </body>
    </html>
    """

def test_scrape_comments(mock_html_response):
    # Mock the requests.get call
    with mock.patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = mock_html_response
        
        scraper = CommentScraper(model_path='./Phi-3-mini-4k-instruct-q4.gguf')
        comments = scraper.scrape_comments("http://mock-url.com")
        
        assert len(comments) == 3
        assert comments == [
            "Great product!",
            "Very satisfied with my purchase.",
            "Not as expected."
        ]
    
def test_get_sentiment():
    scraper = CommentScraper(model_path='./Phi-3-mini-4k-instruct-q4.gguf')

    # Test positive sentiment
    sentiment = scraper.get_sentiment("This is a great product!")
    print(sentiment)  # Check what the model returns

    # Test negative sentiment
    sentiment = scraper.get_sentiment("This is a terrible product!")
    print(sentiment)

    # Test neutral sentiment
    sentiment = scraper.get_sentiment("It's an okay product.")
    print(sentiment)


def test_save_comments_with_sentiment():
    scraper = CommentScraper(model_path='./Phi-3-mini-4k-instruct-q4.gguf')
    comments = ["Great product!", "Very satisfied with my purchase."]
    sentiments = ["positive", "positive"]
    
    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        scraper.save_comments_with_sentiment(comments, sentiments, "test_comments.txt")
        
        # Check if the file was opened for writing
        mock_file.assert_called_with("test_comments.txt", "w", encoding="utf-8")
        
        # Check if the correct data was written to the file
        mock_file().write.assert_any_call("Comment: Great product!\nSentiment: positive\n\n")
        mock_file().write.assert_any_call("Comment: Very satisfied with my purchase.\nSentiment: positive\n\n")

def test_read_urls_from_file():
    # Mocking the file reading
    with mock.patch("builtins.open", mock.mock_open(read_data="http://mock-url.com\nhttp://another-mock-url.com\n")):
        scraper = CommentScraper(model_path='./Phi-3-mini-4k-instruct-q4.gguf')
        urls = scraper.read_urls_from_file("Links.txt")
        
        assert urls == ["http://mock-url.com", "http://another-mock-url.com"]

def test_create_sentiment_graph():
    scraper = CommentScraper(model_path='./Phi-3-mini-4k-instruct-q4.gguf')
    
    urls = ["http://mock-url.com", "http://another-mock-url.com"]
    all_sentiments = [
        ["positive", "negative", "neutral"],
        ["positive", "positive", "positive"]
    ]
    url_to_filename_mapping = {
        "http://mock-url.com": "mock_url_comments.txt",
        "http://another-mock-url.com": "another_mock_url_comments.txt"
    }

    with mock.patch("matplotlib.pyplot.savefig") as mock_savefig:
        scraper.create_sentiment_graph(urls, all_sentiments, url_to_filename_mapping)
        
        # Check if the graph was saved
        mock_savefig.assert_called_once_with("combined_sentiment_graph.png")
