from playwright.sync_api import sync_playwright
import time
import logging
import yaml
import os
import re

# Ensure the logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "app.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class Scraper:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            raise

        self.base_interval = self.config['scraping']['base_interval']
        self.max_interval = self.config['scraping']['max_interval']
        self.min_numbers_per_fetch = self.config['scraping']['min_numbers_per_fetch']
        self.current_interval = self.base_interval
        self.url = self.config['aviator']['url']
        self.login_required = self.config['aviator']['login_required']
        self.login_url = self.config['aviator']['login_url']
        self.username = self.config['aviator']['username']
        self.password = self.config['aviator']['password']
        self.username_field = self.config['aviator']['username_field']
        self.password_field = self.config['aviator']['password_field']
        self.login_button = self.config['aviator']['login_button']

        # Updated selectors based on the new HTML structure
        self.parent_selectors = [
            'div.payouts.ng-star-inserted',
            'div.payouts-block',
            'div.payouts',
            'div.payout',
            'div.history-block',
            'body'
        ]
        self.target_selectors = [
            'app-bubble-multiplier div.bubble-multiplier.font-weight-bold',
            'div[appcoloredmultiplier].bubble-multiplier.font-weight-bold',
            'div.bubble-multiplier.font-weight-bold',
            'div.bubble-multiplier',
            'div[class*="multiplier"]'
        ]

    def _login(self, page):
        try:
            page.goto(self.login_url, timeout=60000)
            page.fill(self.username_field, self.username)
            page.fill(self.password_field, self.password)
            page.click(self.login_button)
            page.wait_for_timeout(5000)  # Wait for login to complete
            logging.info("Login successful.")
        except Exception as e:
            logging.error(f"Error during login: {e}")
            raise

    def scrape_numbers(self, processed_numbers: set) -> list:
        numbers = []
        retries = 3
        backoff = 5

        for attempt in range(retries):
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context()
                    page = context.new_page()

                    if self.login_required:
                        self._login(page)

                    page.goto(self.url, timeout=60000)
                    logging.info(f"Navigated to {self.url}")

                    # Wait for the page to load basic content
                    page.wait_for_load_state("domcontentloaded")
                    time.sleep(2)  # Small delay for initial render

                    # Try to click a history button to load dynamic content
                    history_buttons = [
                        'button.history-btn',
                        'button.show-history',
                        'div.previous-rounds',
                        'a.history-link',
                        'button[aria-label="Show History"]'
                    ]
                    for btn_selector in history_buttons:
                        try:
                            page.click(btn_selector, timeout=5000)
                            logging.info(f"Clicked history button with selector: {btn_selector}")
                            time.sleep(2)  # Wait for history to load
                            break
                        except Exception as e:
                            logging.debug(f"History button {btn_selector} not found: {e}")

                    # Wait for any parent container
                    parent_found = False
                    for selector in self.parent_selectors:
                        try:
                            page.wait_for_selector(selector, state="attached", timeout=30000)
                            logging.info(f"Parent container {selector} detected.")
                            parent_found = True
                            break
                        except Exception as e:
                            logging.warning(f"Parent selector {selector} not found: {e}")
                    if not parent_found:
                        logging.warning("No specific parent container found, proceeding with body.")

                    # Scroll multiple times to load dynamic content
                    for _ in range(3):
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        time.sleep(2)  # Wait for content to load after each scroll

                    # Wait for target elements
                    max_wait = 60000  # 60 seconds total wait
                    wait_interval = 5000  # 5-second intervals
                    elements_found = False

                    for _ in range(max_wait // wait_interval):
                        for selector in self.target_selectors:
                            elements = page.query_selector_all(selector)
                            if elements:
                                elements_found = True
                                logging.info(f"Found {len(elements)} elements with selector: {selector}")
                                break
                        if elements_found:
                            break
                        time.sleep(wait_interval / 1000)
                    if not elements_found:
                        # Log page HTML for debugging
                        html = page.content()
                        with open(os.path.join(logs_dir, f"page_dump_attempt_{attempt+1}.html"), 'w', encoding='utf-8') as f:
                            f.write(html)
                        logging.warning(f"Could not find elements with any selector after {max_wait/1000} seconds. Page HTML dumped to logs/page_dump_attempt_{attempt+1}.html")
                        context.close()
                        browser.close()
                        return []

                    # Extract numbers
                    all_numbers = []
                    for element in elements:
                        text = element.inner_text().strip()
                        match = re.search(r'(\d+\.?\d*)', text)
                        if match:
                            num = float(match.group(1))
                            if 1.0 <= num <= 200.0 and num not in processed_numbers:
                                all_numbers.append(num)

                    numbers = sorted(list(set(all_numbers)), reverse=True)
                    processed_numbers.update(numbers)

                    if len(numbers) >= self.min_numbers_per_fetch:
                        self.current_interval = max(self.base_interval, self.current_interval * 0.8)
                    else:
                        self.current_interval = min(self.max_interval, self.current_interval * 1.2)
                    logging.info(f"Adjusted scraping interval to {self.current_interval:.2f} seconds.")
                    logging.info(f"New numbers fetched: {numbers}")

                    context.close()
                    browser.close()
                    return numbers

            except Exception as e:
                logging.error(f"Scraping attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logging.error("All scraping attempts failed.")
                    if 'context' in locals():
                        context.close()
                    if 'browser' in locals():
                        browser.close()
                    return []

        return numbers