import urllib.request
from html.parser import HTMLParser

class LinkParser(HTMLParser):
    """Parser to extract links from HTML."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        """Handle start tags."""
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    self.links.append(value)

def download_and_parse(url, filename):
    """Download webpage, parse links, and save to file."""
    try:
        # Download webpage
        response = urllib.request.urlopen(url)
        html = response.read().decode('utf-8')

        # Parse links
        parser = LinkParser()
        parser.feed(html)

        # Save links to file
        with open(filename, 'w') as f:
            for link in parser.links:
                f.write(link + '\n')

        print(f"Links saved to {filename}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    url = input("Enter URL: ")
    filename = input("Enter filename: ")
    download_and_parse(url, filename)