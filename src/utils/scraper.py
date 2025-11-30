import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import re
from typing import List, Dict, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ITUWebScraper:
    def __init__(self, base_url: str = "https://itustudent.itu.dk/Your-Programme/Exchange-students/Welcome"):
        self.ALLOWED_PATHS = ['/About-ITU', '/Research', '/Education', '/News', '/Contact', '/About-ITU/Press', '/About-ITU/Press/News-from-ITU']
        self.BLOCKED_PATHS = ['/News','/Event','/Campus-Life']
        self.BLOCKED_CLASSES = []#["skip-main", "nav-brand", "footer", "header", "social", "logo", "nav-brand", "nav-dropdown", "navigation", "sidebar", "social-links", "social-media"]
        self.SOCIAL_DOMAINS =  ["facebook", "instagram", "linkedin","youtube", "bsky", "twitter", "tiktok"]
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def is_social_url(self,url: str) -> bool:
        netloc = urlparse(url).netloc.lower()
        return any(domain in netloc for domain in self.SOCIAL_DOMAINS)
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to ITU domain"""
        parsed = urlparse(url)
        
        # Skip all news URLs
        if url.startswith('https://en.itu.dk/News'):
            return False
        
        if self.is_social_url(url):
            return False
        # Skip news article URLs from the old path
        if '/About-ITU/Press/News-from-ITU/' in url: #and url != 'https://en.itu.dk/About-ITU/Press/News-from-ITU':
            return False

    
        return (
            # parsed.netloc == 'en.itu.dk' and
            not any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js']) and
            '#' not in url and
            'mailto:' not in url and
            'tel:' not in url and
            'https://itustudent' in url and
            'news' not in url.lower() and 
            'study-abroad' not in url.lower()
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common navigation and footer elements
        footer_keywords = [
            'Skip to main content', 'Menu', 'Search', 'Login', 'Contact',
            'Follow us', 'Social media', 'Copyright', 'Privacy policy',
            'Terms of service', 'All rights reserved', 'Cookie policy',
            'Facebook', 'Twitter', 'LinkedIn', 'Instagram', 'YouTube'
        ]
        
        for keyword in footer_keywords:
            text = re.sub(re.escape(keyword), '', text, flags=re.IGNORECASE)
        
        return text
    
    def is_footer_element(self, element) -> bool:
        """Check if an element is part of the footer or navigation"""
        # Check if element has footer-related classes
        element_classes = element.get('class', [])
        if any(cls.lower() in ['footer', 'nav', 'navigation', 'header', 'sidebar', 'social'] for cls in element_classes):
            return True
        
        # Check if element is inside footer-related containers
        parent = element.parent
        while parent:
            parent_classes = parent.get('class', [])
            parent_id = parent.get('id', '')
            
            # Check for footer-related classes and IDs
            if any(cls.lower() in ['footer', 'nav', 'navigation', 'header', 'sidebar', 'social'] for cls in parent_classes):
                return True
            if any(id_part.lower() in ['footer', 'nav', 'navigation', 'header', 'sidebar', 'social'] for id_part in parent_id.split('-')):
                return True
                
            parent = parent.parent
            if parent and parent.name == 'body':  # Stop at body level
                break
        
        return False
    
    def remove_footer_elements(self, soup: BeautifulSoup):
        """Remove footer and navigation elements from the soup"""
        # Common footer and navigation selectors
        footer_selectors = [
            'footer',
            '.footer',
            '#footer',
            'nav[class*="footer"]',
            'div[class*="footer"]',
            '.navigation',
            '.nav',
            '.header',
            '.sidebar',
            '.social',
            '.social-links',
            '.social-media',
            'div[class*="social"]',
            '.skip-to-main',
            '.skip-main',
            '.nav-brand',
            '.nav-dropdown'
        ]
        
        for selector in footer_selectors:
            elements = soup.select(selector)
            for element in elements:
                element.decompose()  # Remove the element completely
        
        # Remove elements with specific classes
        for class_name in self.BLOCKED_CLASSES:
            elements = soup.find_all(class_=re.compile(class_name, re.I))
            for element in elements:
                element.decompose()
    
    def extract_page_content(self, url: str, soup: BeautifulSoup) -> Dict:
        """Extract structured content from a page"""
        content = {
            'url': url,
            'title': '',
            'headings': [],
            'paragraphs': [],
            'links': [],
            'metadata': {}
        }
        
        # Remove footer and navigation elements before extracting content
        self.remove_footer_elements(soup)
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            content['title'] = self.clean_text(title_tag.get_text())
        
        # Extract main headings (h1, h2, h3) - exclude footer headings
        for level in ['h1', 'h2', 'h3']:
            headings = soup.find_all(level)
            for heading in headings:
                # Skip headings in footer areas
                if self.is_footer_element(heading):
                    continue
                    
                heading_text = self.clean_text(heading.get_text())
                if heading_text and len(heading_text) > 3:
                    content['headings'].append({
                        'level': level,
                        'text': heading_text
                    })
        
        # Extract paragraphs - exclude footer paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            # Skip paragraphs in footer areas
            if self.is_footer_element(p):
                continue
                
            paragraph_text = self.clean_text(p.get_text())
            if paragraph_text and len(paragraph_text) > 20:  # Filter out very short paragraphs
                content['paragraphs'].append(paragraph_text)
        
        # Extract links
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            link_text = self.clean_text(link.get_text())
            if href and link_text and len(link_text) > 2:
                full_url = urljoin(url, href)
                if self.is_valid_url(full_url):
                    content['links'].append({
                        'url': full_url,
                        'text': link_text
                    })
        
        # Extract metadata
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            content['metadata']['description'] = meta_description.get('content', '')
        
        # Extract breadcrumbs if available
        breadcrumbs = soup.find_all(['nav', 'ol', 'ul'], class_=re.compile(r'breadcrumb', re.I))
        if breadcrumbs:
            content['metadata']['breadcrumbs'] = [self.clean_text(bc.get_text()) for bc in breadcrumbs]
        
        return content
    
    def scrape_page(self, url: str) -> Dict:
        """Scrape a single page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content = self.extract_page_content(url, soup)
            
            # Add full text content for embedding
            full_text = f"{content['title']}\n\n"
            for heading in content['headings']:
                full_text += f"{heading['text']}\n"
            for paragraph in content['paragraphs']:
                full_text += f"{paragraph}\n"
            
            content['full_text'] = self.clean_text(full_text)
            content['word_count'] = len(content['full_text'].split())
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    

    def is_relevant(self, url):
        """Check if URL is relevant"""
        return not any(path in url for path in self.BLOCKED_PATHS)

    def fetch_url(self, url: str) -> Set[str]:
        try:

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            links = set()
            print("len", len(soup.find_all('a', href=True)))
            
            
            for link in soup.find_all('a', href=True): # get all hyperlink objects
                 link_class = link.get("class", [])
                 
                 # skip footer and navigation links
                #  if self.is_footer_element(link):
                #      continue
                 
                 # skip irrelevant classes
                 if any(cls in link_class for cls in self.BLOCKED_CLASSES):
                     continue

                 href = link.get('href') # get the link to next page
                 if href.lower().endswith((".pdf", ".ashx", ".jpg", ".png", ".zip")):
                     continue

                 full_url = urljoin(url, href)
             
                 if self.is_valid_url(full_url) and self.is_relevant(full_url):
                    links.add(full_url)

            logger.info(f"Found {len(links)} links from {url}")
            
            return links
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return set()
    
    def discover_urls(self, start_url: str) -> Set[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        urls_to_visit = {start_url}
        discovered_urls = set()

        with ThreadPoolExecutor(max_workers=10) as executor:
            logger.info(f"Starting to discover URLs from {start_url}")
            while urls_to_visit:
                futures = {executor.submit(self.fetch_url, url): url for url in urls_to_visit}
                print("All fuutress")
                print(futures)
                
                

                urls_to_visit = set() 
                for future in as_completed(futures):
                    links = future.result()
                    new_links = links - self.visited_urls - discovered_urls
                    new_links = list(set(new_links))
                    print('new links set', len(new_links))
                
                    
                    for the_link in new_links:
                        print(the_link)
                    
                   
                    discovered_urls.update(new_links)
                    urls_to_visit.update(new_links)
                    self.visited_urls.update(links)
                    print("All visited urls")
                    print(self.visited_urls)
                    print("urls to visit")
                    print(urls_to_visit)
                    
            

        return discovered_urls

    def scrape_all_pages(self, max_pages: int = 100) -> List[Dict]:
        """Scrape all pages from the ITU website"""
        logger.info("Starting to discover URLs...")
        all_urls = self.discover_urls(self.base_url)
        # Limit the number of pages to scrape (if max_pages is specified)
        if max_pages is not None:
            urls_to_scrape = list(all_urls)[:max_pages]
            logger.info(f"Found {len(all_urls)} URLs, will scrape {len(urls_to_scrape)} pages")
        else:
            urls_to_scrape = list(all_urls)
            logger.info(f"Found {len(all_urls)} URLs, will scrape ALL pages")
    
        
        scraped_data = []
        
        for url in tqdm(urls_to_scrape, desc="Scraping pages"):
            content = self.scrape_page(url)
            print(f"Content for url {url}")
            print(True if content and content['full_text'] else False)

            if content and content['full_text']:
                
                scraped_data.append(content)
                logger.info(f"Scraped: {content['title'][:50]}... ({content['word_count']} words)")
            
            # Be respectful - add delay between requests
            time.sleep(1)
        
        self.scraped_data = scraped_data
        return scraped_data
    
    def save_to_json(self, filename: str = None):
        """Save scraped data to JSON file"""
        if filename is None:
            filename = os.path.join('data', 'vectors', 'itu_scraped_data.json')
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.scraped_data)} pages to {filename}")
    
    def save_urls_list(self, filename: str = None):
        """Save list of all scraped URLs to a text file"""
        try:
            if filename is None:
                filename = os.path.join('data', 'vectors', 'itu_scraped_urls.txt')
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            urls = [page['url'] for page in self.scraped_data]
            with open(filename, 'w', encoding='utf-8') as f:
                for url in sorted(urls):
                    f.write(f"{url}\n")
            logger.info(f"Saved {len(urls)} URLs to {filename}")
        except Exception as e:
            logger.error(f"Error saving URLs list: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about scraped data"""
        if not self.scraped_data:
            return {}
        
        total_pages = len(self.scraped_data)
        total_words = sum(page['word_count'] for page in self.scraped_data)
        avg_words = total_words / total_pages if total_pages > 0 else 0
        
        return {
            'total_pages': total_pages,
            'total_words': total_words,
            'average_words_per_page': avg_words,
            'urls_visited': len(self.visited_urls)
        }

def main():
    """Main function to run the scraper"""
    scraper = ITUWebScraper()
    
    print("🚀 Starting ITU website scraper...")
    print("📋 This will scrape all pages from https://en.itu.dk")
    print("⏱️  This may take several minutes...")
    
    # Scrape all pages
    scraped_data = scraper.scrape_all_pages(max_pages=30)  # Limit to 30 pages for demo
    logger.info(f"Scraped {len(scraped_data)} pages")
    
    # Save data
    scraper.save_to_json()
    
    # Print statistics
    stats = scraper.get_statistics()
    print("\n📊 Scraping Statistics:")
    print(f"   Total pages scraped: {stats['total_pages']}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Average words per page: {stats['average_words_per_page']:.1f}")
    print(f"   URLs visited: {stats['urls_visited']}")
    
    print("\n✅ Scraping completed! Data saved to 'itu_scraped_data.json'")

if __name__ == "__main__":
    main()
