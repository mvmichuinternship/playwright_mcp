"""
Enhanced Playwright MCP Server for Dual Model Browser Automation
Optimized for Core LLM + Moondream Agent coordination
"""

import asyncio
import base64
import json
import logging
from typing import List, Optional, Union, Dict, Tuple
import os
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, TextContent, ImageContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global browser state
playwright_instance = None
browser: Optional[Browser] = None
context: Optional[BrowserContext] = None
page: Optional[Page] = None
last_screenshot_b64: Optional[str] = None

# Initialize MCP Server
mcp = FastMCP("enhanced-playwright-server")

async def ensure_browser() -> Page:
    """Ensure browser, context, and page are initialized"""
    global playwright_instance, browser, context, page

    if not playwright_instance:
        playwright_instance = await async_playwright().start()

    if not browser:
        browser = await playwright_instance.chromium.launch(
            headless=False,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'  # Better for screenshots
            ]
        )

    if not context:
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9'
            }
        )

    if not page:
        page = await context.new_page()
        # Disable animations for more stable screenshots
        await page.add_init_script("""
            // Disable CSS animations and transitions
            const css = `
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-delay: -0.01ms !important;
                    transition-duration: 0.01ms !important;
                    transition-delay: -0.01ms !important;
                }
            `;
            const style = document.createElement('style');
            style.appendChild(document.createTextNode(css));
            document.head.appendChild(style);
        """)

    return page

@mcp.tool()
async def take_screenshot_and_store(
    full_page: bool = False,
    format: str = 'png',
    quality: int = 90
) -> List[Union[TextContent, ImageContent]]:
    """
    Take a high-quality screenshot optimized for Moondream analysis
    This is the PRIMARY tool for the screenshot-first workflow
    """
    global last_screenshot_b64

    try:
        current_page = await ensure_browser()

        # Wait for page to be stable
        await current_page.wait_for_load_state('networkidle', timeout=10000)
        await asyncio.sleep(0.5)  # Additional stability wait

        screenshot_options = {
            'type': format,
            'full_page': full_page,
        }

        if format == 'jpeg':
            screenshot_options['quality'] = quality

        screenshot_bytes = await current_page.screenshot(**screenshot_options)
        last_screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # Get comprehensive page context
        page_url = current_page.url
        page_title = await current_page.title()
        viewport = current_page.viewport_size

        result = {
            'success': True,
            'format': format,
            'full_page': full_page,
            'quality': quality if format == 'jpeg' else None,
            'page_url': page_url,
            'page_title': page_title,
            'viewport': viewport,
            'screenshot_size': len(screenshot_bytes),
            'timestamp': asyncio.get_event_loop().time(),
            'message': 'High-quality screenshot captured and ready for Moondream analysis'
        }

        return [
            TextContent(type="text", text=json.dumps(result, indent=2)),
            ImageContent(
                type="image",
                data=last_screenshot_b64,
                mimeType=f"image/{format}"
            )
        ]

    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'message': 'Failed to capture screenshot'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def click_at_coordinates(
    x: int,
    y: int,
    button: str = 'left',
    delay: int = 100,
    click_count: int = 1
) -> List[TextContent]:
    """
    Click at specific coordinates with enhanced feedback
    Used by smart tools after Moondream provides coordinates
    """
    try:
        current_page = await ensure_browser()

        # Validate coordinates are within viewport
        viewport = current_page.viewport_size
        if not (0 <= x <= viewport['width'] and 0 <= y <= viewport['height']):
            logger.warning(f"Coordinates ({x}, {y}) may be outside viewport {viewport}")

        # Perform the click with enhanced options
        await current_page.mouse.click(
            x, y,
            button=button,
            delay=delay,
            click_count=click_count
        )

        # Wait for potential page changes and UI updates
        try:
            await current_page.wait_for_load_state('networkidle', timeout=3000)
        except:
            pass  # Continue even if networkidle times out

        await asyncio.sleep(0.5)  # Additional wait for UI updates

        # Get updated page info
        new_url = current_page.url
        new_title = await current_page.title()

        result = {
            'success': True,
            'coordinates': {'x': x, 'y': y},
            'button': button,
            'delay': delay,
            'click_count': click_count,
            'new_url': new_url,
            'new_title': new_title,
            'viewport': current_page.viewport_size,
            'timestamp': asyncio.get_event_loop().time(),
            'message': f'Successfully clicked at coordinates ({x}, {y}) with {button} button'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Click at coordinates failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'coordinates': {'x': x, 'y': y},
            'message': f'Failed to click at coordinates ({x}, {y})'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def type_text_at_coordinates(
    x: int,
    y: int,
    text: str,
    clear_first: bool = True,
    delay: int = 50
) -> List[TextContent]:
    """
    Click at coordinates and type text
    Used by smart input tools after Moondream provides coordinates
    """
    try:
        current_page = await ensure_browser()

        # Click to focus the input field
        await current_page.mouse.click(x, y, delay=100)
        await asyncio.sleep(0.3)

        # Clear existing content if requested
        if clear_first:
            # Select all and delete
            await current_page.keyboard.press('Control+a')
            await asyncio.sleep(0.1)
            await current_page.keyboard.press('Delete')
            await asyncio.sleep(0.2)

        # Type the text
        if delay > 0:
            # Type with delay between characters for more natural input
            await current_page.keyboard.type(text, delay=delay)
        else:
            # Fast typing
            await current_page.keyboard.type(text)

        # Wait for any input validation or UI updates
        await asyncio.sleep(0.3)

        result = {
            'success': True,
            'coordinates': {'x': x, 'y': y},
            'text': text,
            'text_length': len(text),
            'cleared_first': clear_first,
            'delay': delay,
            'timestamp': asyncio.get_event_loop().time(),
            'message': f'Successfully typed "{text}" at coordinates ({x}, {y})'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Type text at coordinates failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'coordinates': {'x': x, 'y': y},
            'text': text,
            'message': f'Failed to type text at coordinates ({x}, {y})'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def launch_browser(
    headless: bool = False,
    browser_type: str = 'chromium',
    viewport_width: int = 1280,
    viewport_height: int = 720,
    user_agent: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    slow_mo: int = 0,
    devtools: bool = False
) -> List[TextContent]:
    """Launch browser with custom configuration"""
    global playwright_instance, browser, context, page

    try:
        # Close existing browser if running
        if browser:
            await close_browser()

        # Start playwright
        if not playwright_instance:
            playwright_instance = await async_playwright().start()

        # Prepare browser args
        browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
        if extra_args:
            browser_args.extend(extra_args)

        # Launch browser based on type
        if browser_type in ['chrome', 'google-chrome', 'chromium']:
            browser = await playwright_instance.chromium.launch(
                headless=headless,
                args=browser_args,
                slow_mo=slow_mo,
                devtools=devtools
            )
        elif browser_type.lower() == 'firefox':
            browser = await playwright_instance.firefox.launch(
                headless=headless,
                args=browser_args,
                slow_mo=slow_mo,
                devtools=devtools
            )
        elif browser_type.lower() == 'webkit':
            browser = await playwright_instance.webkit.launch(
                headless=headless,
                args=browser_args,
                slow_mo=slow_mo,
                devtools=devtools
            )
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")

        # Set up default user agent if not provided
        if not user_agent:
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

        # Create context
        context = await browser.new_context(
            viewport={'width': viewport_width, 'height': viewport_height},
            user_agent=user_agent
        )

        # Create page
        page = await context.new_page()

        result = {
            'success': True,
            'browser_type': browser_type,
            'headless': headless,
            'viewport': {'width': viewport_width, 'height': viewport_height},
            'user_agent': user_agent,
            'slow_mo': slow_mo,
            'devtools': devtools,
            'message': f'{browser_type.title()} browser launched successfully'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def navigate(
    url: str,
    wait_until: str = 'networkidle',
    timeout: int = 30000
) -> List[TextContent]:
    """
    Navigate to a URL with enhanced stability for screenshot workflow
    """
    global last_screenshot_b64

    try:
        current_page = await ensure_browser()

        # Clear screenshot cache since we're navigating
        last_screenshot_b64 = None

        # Navigate with comprehensive waiting
        response = await current_page.goto(
            url,
            wait_until=wait_until,
            timeout=timeout
        )

        # Additional wait for dynamic content
        await asyncio.sleep(1.0)

        # Try to wait for any remaining network activity
        try:
            await current_page.wait_for_load_state('networkidle', timeout=5000)
        except:
            pass

        # Get comprehensive page information
        page_info = {
            'success': True,
            'url': current_page.url,
            'title': await current_page.title(),
            'status': response.status if response else None,
            'viewport': current_page.viewport_size,
            'timestamp': asyncio.get_event_loop().time(),
            'screenshot_invalidated': True,
            'message': 'Navigation completed successfully - screenshot cache cleared'
        }

        return [TextContent(type="text", text=json.dumps(page_info, indent=2))]

    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'url': url,
            'message': f'Failed to navigate to {url}'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def scroll_page(
    direction: str = 'down',
    amount: int = 3,
    smooth: bool = False
) -> List[TextContent]:
    """
    Scroll the page with screenshot cache invalidation
    """
    global last_screenshot_b64

    try:
        current_page = await ensure_browser()

        # Clear screenshot cache since we're scrolling
        last_screenshot_b64 = None

        # Scroll actions
        scroll_actions = {
            'down': lambda: current_page.keyboard.press('ArrowDown'),
            'up': lambda: current_page.keyboard.press('ArrowUp'),
            'page_down': lambda: current_page.keyboard.press('PageDown'),
            'page_up': lambda: current_page.keyboard.press('PageUp'),
            'home': lambda: current_page.keyboard.press('Home'),
            'end': lambda: current_page.keyboard.press('End')
        }

        scroll_action = scroll_actions.get(direction, scroll_actions['down'])

        for i in range(amount):
            await scroll_action()
            await asyncio.sleep(0.2 if smooth else 0.1)

        # Wait for scroll to complete
        await asyncio.sleep(0.5)

        result = {
            'success': True,
            'direction': direction,
            'amount': amount,
            'smooth': smooth,
            'timestamp': asyncio.get_event_loop().time(),
            'screenshot_invalidated': True,
            'message': f'Scrolled {direction} {amount} times - screenshot cache cleared'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Scroll failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'direction': direction,
            'amount': amount,
            'message': f'Failed to scroll {direction}'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def press_key(
    key: str,
    modifiers: List[str] = None
) -> List[TextContent]:
    """
    Press keyboard key(s) with modifier support
    """
    try:
        current_page = await ensure_browser()

        if modifiers:
            # Handle key combinations
            modifier_str = '+'.join(modifiers + [key])
            await current_page.keyboard.press(modifier_str)
        else:
            # Single key press
            await current_page.keyboard.press(key)

        await asyncio.sleep(0.1)

        result = {
            'success': True,
            'key': key,
            'modifiers': modifiers or [],
            'timestamp': asyncio.get_event_loop().time(),
            'message': f'Key "{key}" pressed successfully' + (f' with modifiers {modifiers}' if modifiers else '')
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Key press failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'key': key,
            'modifiers': modifiers or [],
            'message': f'Failed to press key "{key}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def get_page_info() -> List[TextContent]:
    """Get comprehensive current page information"""
    try:
        current_page = await ensure_browser()

        # Get page metrics
        try:
            metrics = await current_page.evaluate("""
                () => ({
                    documentHeight: document.documentElement.scrollHeight,
                    documentWidth: document.documentElement.scrollWidth,
                    viewportHeight: window.innerHeight,
                    viewportWidth: window.innerWidth,
                    scrollX: window.scrollX,
                    scrollY: window.scrollY,
                    readyState: document.readyState,
                    activeElement: document.activeElement ? {
                        tagName: document.activeElement.tagName,
                        id: document.activeElement.id,
                        className: document.activeElement.className
                    } : null
                })
            """)
        except:
            metrics = {}

        result = {
            'success': True,
            'url': current_page.url,
            'title': await current_page.title(),
            'viewport': current_page.viewport_size,
            'metrics': metrics,
            'has_stored_screenshot': last_screenshot_b64 is not None,
            'screenshot_timestamp': asyncio.get_event_loop().time() if last_screenshot_b64 else None,
            'timestamp': asyncio.get_event_loop().time(),
            'message': 'Page information retrieved successfully'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Get page info failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'message': 'Failed to get page information'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def wait_for_element(
    selector: str = None,
    text: str = None,
    timeout: int = 10000
) -> List[TextContent]:
    """
    Wait for an element to appear (useful before taking screenshot)
    """
    try:
        current_page = await ensure_browser()

        if selector:
            await current_page.wait_for_selector(selector, timeout=timeout)
            message = f'Element with selector "{selector}" appeared'
        elif text:
            await current_page.wait_for_function(
                f'document.body && document.body.innerText.includes("{text}")',
                timeout=timeout
            )
            message = f'Text "{text}" appeared on page'
        else:
            await current_page.wait_for_load_state('networkidle', timeout=timeout)
            message = 'Page reached networkidle state'

        result = {
            'success': True,
            'selector': selector,
            'text': text,
            'timeout': timeout,
            'timestamp': asyncio.get_event_loop().time(),
            'message': message
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Wait for element failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'selector': selector,
            'text': text,
            'timeout': timeout,
            'message': 'Failed to wait for element/condition'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def close_browser() -> List[TextContent]:
    """Close browser and clean up all resources"""
    global playwright_instance, browser, context, page, last_screenshot_b64

    try:
        cleanup_steps = []

        if page:
            await page.close()
            page = None
            cleanup_steps.append("Page closed")

        if context:
            await context.close()
            context = None
            cleanup_steps.append("Context closed")

        if browser:
            await browser.close()
            browser = None
            cleanup_steps.append("Browser closed")

        if playwright_instance:
            await playwright_instance.stop()
            playwright_instance = None
            cleanup_steps.append("Playwright stopped")

        # Clear screenshot cache
        last_screenshot_b64 = None
        cleanup_steps.append("Screenshot cache cleared")

        result = {
            'success': True,
            'cleanup_steps': cleanup_steps,
            'timestamp': asyncio.get_event_loop().time(),
            'message': 'Browser closed and all resources cleaned up successfully'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Browser cleanup failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'message': 'Failed to close browser cleanly'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

# Additional utility tools for enhanced functionality

@mcp.tool()
async def get_element_info(
    x: int,
    y: int
) -> List[TextContent]:
    """Get information about element at specific coordinates"""
    try:
        current_page = await ensure_browser()

        # Get element information at coordinates
        element_info = await current_page.evaluate(f"""
            (x, y) => {{
                const element = document.elementFromPoint(x, y);
                if (!element) return null;

                return {{
                    tagName: element.tagName,
                    id: element.id || null,
                    className: element.className || null,
                    textContent: element.textContent ? element.textContent.substring(0, 100) : null,
                    attributes: Object.fromEntries(
                        Array.from(element.attributes).map(attr => [attr.name, attr.value])
                    ),
                    boundingRect: element.getBoundingClientRect(),
                    isClickable: element.onclick !== null ||
                                element.tagName === 'BUTTON' ||
                                element.tagName === 'A' ||
                                element.tagName === 'INPUT' ||
                                element.role === 'button',
                    isInput: element.tagName === 'INPUT' ||
                            element.tagName === 'TEXTAREA' ||
                            element.contentEditable === 'true'
                }}
            }}
        """, x, y)

        result = {
            'success': True,
            'coordinates': {'x': x, 'y': y},
            'element_info': element_info,
            'timestamp': asyncio.get_event_loop().time(),
            'message': f'Element information retrieved for coordinates ({x}, {y})'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Get element info failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'coordinates': {'x': x, 'y': y},
            'message': f'Failed to get element info at ({x}, {y})'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def highlight_coordinates(
    x: int,
    y: int,
    duration: int = 2000,
    color: str = 'red'
) -> List[TextContent]:
    """Highlight specific coordinates on the page (useful for debugging)"""
    try:
        current_page = await ensure_browser()

        # Add a highlight dot at coordinates
        await current_page.evaluate(f"""
            (x, y, duration, color) => {{
                const dot = document.createElement('div');
                dot.style.cssText = `
                    position: fixed;
                    left: ${{x - 10}}px;
                    top: ${{y - 10}}px;
                    width: 20px;
                    height: 20px;
                    background: ${{color}};
                    border: 2px solid white;
                    border-radius: 50%;
                    z-index: 10000;
                    pointer-events: none;
                    opacity: 0.8;
                    animation: pulse 1s infinite;
                `;

                // Add pulse animation
                if (!document.getElementById('highlight-style')) {{
                    const style = document.createElement('style');
                    style.id = 'highlight-style';
                    style.textContent = `
                        @keyframes pulse {{
                            0% {{ transform: scale(1); opacity: 0.8; }}
                            50% {{ transform: scale(1.5); opacity: 0.4; }}
                            100% {{ transform: scale(1); opacity: 0.8; }}
                        }}
                    `;
                    document.head.appendChild(style);
                }}

                document.body.appendChild(dot);

                setTimeout(() => {{
                    if (dot.parentNode) {{
                        dot.parentNode.removeChild(dot);
                    }}
                }}, duration);
            }}
        """, x, y, duration, color)

        result = {
            'success': True,
            'coordinates': {'x': x, 'y': y},
            'duration': duration,
            'color': color,
            'timestamp': asyncio.get_event_loop().time(),
            'message': f'Highlighted coordinates ({x}, {y}) for {duration}ms'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Highlight coordinates failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'coordinates': {'x': x, 'y': y},
            'message': f'Failed to highlight coordinates ({x}, {y})'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def take_element_screenshot(
    x: int,
    y: int,
    width: int = 200,
    height: int = 200
) -> List[Union[TextContent, ImageContent]]:
    """Take a focused screenshot of a specific area (useful for element verification)"""
    try:
        current_page = await ensure_browser()

        # Calculate clip area
        clip_area = {
            'x': max(0, x - width // 2),
            'y': max(0, y - height // 2),
            'width': width,
            'height': height
        }

        screenshot_bytes = await current_page.screenshot(
            type='png',
            clip=clip_area
        )

        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        result = {
            'success': True,
            'center_coordinates': {'x': x, 'y': y},
            'clip_area': clip_area,
            'screenshot_size': len(screenshot_bytes),
            'timestamp': asyncio.get_event_loop().time(),
            'message': f'Element screenshot captured around coordinates ({x}, {y})'
        }

        return [
            TextContent(type="text", text=json.dumps(result, indent=2)),
            ImageContent(
                type="image",
                data=screenshot_b64,
                mimeType="image/png"
            )
        ]

    except Exception as e:
        logger.error(f"Element screenshot failed: {e}")
        error_result = {
            'success': False,
            'error': str(e),
            'coordinates': {'x': x, 'y': y},
            'message': f'Failed to take element screenshot at ({x}, {y})'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def analyze_page_content(
    question: str = "Describe this webpage and list all interactive elements"
) -> List[Union[TextContent, ImageContent]]:
    """
    Analyze webpage content using AI vision model
    This tool acts as a coordination trigger for the dual model client to route analysis to Moondream

    Args:
        question: Specific analysis question for the page content

    Returns:
        Coordination trigger response that will be intercepted by the dual model client
    """
    try:
        current_page = await ensure_browser()

        # Get comprehensive page context for Moondream
        try:
            # Basic page information
            page_title = await current_page.title()
            page_url = current_page.url
            viewport = current_page.viewport_size

            # Get page metrics and element counts
            page_metrics = await current_page.evaluate("""
                () => {
                    const body = document.body;
                    const html = document.documentElement;

                    return {
                        // Page dimensions
                        scrollWidth: Math.max(body.scrollWidth, html.scrollWidth),
                        scrollHeight: Math.max(body.scrollHeight, html.scrollHeight),
                        clientWidth: html.clientWidth,
                        clientHeight: html.clientHeight,
                        scrollTop: window.pageYOffset,
                        scrollLeft: window.pageXOffset,

                        // Element counts for context
                        buttons: document.querySelectorAll('button, input[type="button"], input[type="submit"], [role="button"]').length,
                        links: document.querySelectorAll('a[href]').length,
                        inputs: document.querySelectorAll('input, textarea, select').length,
                        images: document.querySelectorAll('img').length,
                        forms: document.querySelectorAll('form').length,
                        headings: document.querySelectorAll('h1, h2, h3, h4, h5, h6').length,

                        // Page state
                        loaded: document.readyState,
                        hasTitle: !!document.title,
                        lang: document.documentElement.lang || 'unknown'
                    };
                }
            """)

            page_context = {
                "title": page_title,
                "url": page_url,
                "viewport": viewport,
                "metrics": page_metrics,
                "timestamp": asyncio.get_event_loop().time()
            }

        except Exception as context_error:
            logger.warning(f"Could not gather full page context: {context_error}")
            page_context = {
                "title": "Unknown",
                "url": current_page.url,
                "viewport": current_page.viewport_size,
                "error": str(context_error)
            }

        # Create coordination trigger response
        # The dual model client will intercept this and route to Moondream
        coordination_response = {
            "success": True,
            "tool": "analyze_page_content",
            "question": question,
            "page_context": page_context,
            "coordination_trigger": True,
            "status": "ready_for_moondream_analysis",
            "message": f"Page analysis request prepared for Moondream: '{question}'",
            "instructions_for_client": {
                "action": "route_to_moondream_agent",
                "analysis_type": "page_content_analysis",
                "question": question,
                "context": page_context,
                "requires_screenshot": True
            },
            "workflow_note": "This response triggers dual model coordination - client will route to Moondream for visual analysis"
        }

        return [
            TextContent(
                type="text",
                text=json.dumps(coordination_response, indent=2)
            )
        ]

    except Exception as e:
        logger.error(f"Error in analyze_page_content tool: {e}")

        error_response = {
            "success": False,
            "tool": "analyze_page_content",
            "error": str(e),
            "question": question,
            "coordination_status": "FAILED - Exception in MCP tool execution",
            "message": "Failed to prepare page analysis request"
        }

        return [
            TextContent(
                type="text",
                text=json.dumps(error_response, indent=2)
            )
        ]

# Add these tools to your enhanced_playwright_server.py

@mcp.tool()
async def smart_click_element(
    element_description: str,
    button: str = 'left',
    delay: int = 100,
    click_count: int = 1,
    timeout: int = 10000
) -> List[TextContent]:
    """
    Smart click on an element using natural language description
    This tool coordinates with Moondream agent for precise coordinate detection

    Args:
        element_description: Natural language description of the element to click (e.g., "login button", "search box", "submit button")
        button: Mouse button to use ('left', 'right', 'middle')
        delay: Delay between mousedown and mouseup in milliseconds
        click_count: Number of clicks (1 for single, 2 for double click)
        timeout: Maximum time to wait for element detection in milliseconds
    """
    try:
        current_page = await ensure_browser()

        # This is a coordination trigger that will be intercepted by the dual model client
        # The actual coordinate detection will be handled by the Moondream agent

        result = {
            'success': True,
            'tool': 'smart_click_element',
            'element_description': element_description,
            'button': button,
            'delay': delay,
            'click_count': click_count,
            'timeout': timeout,
            'coordination_trigger': True,
            'status': 'ready_for_coordinate_detection',
            'message': f'Smart click request prepared for element: "{element_description}"',
            'instructions_for_client': {
                'action': 'coordinate_detection_required',
                'interaction_type': 'click',
                'element_description': element_description,
                'click_parameters': {
                    'button': button,
                    'delay': delay,
                    'click_count': click_count
                },
                'requires_screenshot': True
            },
            'workflow_note': 'This response triggers dual model coordination - client will use Moondream for coordinate detection'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Smart click preparation failed: {e}")
        error_result = {
            'success': False,
            'tool': 'smart_click_element',
            'error': str(e),
            'element_description': element_description,
            'message': f'Failed to prepare smart click for element: "{element_description}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def smart_input_text(
    element_description: str,
    text: str,
    clear_first: bool = True,
    delay: int = 50,
    press_enter: bool = False,
    timeout: int = 10000
) -> List[TextContent]:
    """
    Smart text input into an element using natural language description
    This tool coordinates with Moondream agent for precise coordinate detection

    Args:
        element_description: Natural language description of the input field (e.g., "username field", "search box", "email input")
        text: Text to type into the field
        clear_first: Whether to clear existing content before typing
        delay: Delay between keystrokes in milliseconds (0 for no delay)
        press_enter: Whether to press Enter after typing the text
        timeout: Maximum time to wait for element detection in milliseconds
    """
    try:
        current_page = await ensure_browser()

        # This is a coordination trigger that will be intercepted by the dual model client
        # The actual coordinate detection will be handled by the Moondream agent

        result = {
            'success': True,
            'tool': 'smart_input_text',
            'element_description': element_description,
            'text': text,
            'clear_first': clear_first,
            'delay': delay,
            'press_enter': press_enter,
            'timeout': timeout,
            'coordination_trigger': True,
            'status': 'ready_for_coordinate_detection',
            'message': f'Smart input request prepared for element: "{element_description}" with text: "{text}"',
            'instructions_for_client': {
                'action': 'coordinate_detection_required',
                'interaction_type': 'input',
                'element_description': element_description,
                'input_parameters': {
                    'text': text,
                    'clear_first': clear_first,
                    'delay': delay,
                    'press_enter': press_enter
                },
                'requires_screenshot': True
            },
            'workflow_note': 'This response triggers dual model coordination - client will use Moondream for coordinate detection'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Smart input preparation failed: {e}")
        error_result = {
            'success': False,
            'tool': 'smart_input_text',
            'error': str(e),
            'element_description': element_description,
            'text': text,
            'message': f'Failed to prepare smart input for element: "{element_description}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def smart_scroll_to_element(
    element_description: str,
    alignment: str = 'center',
    timeout: int = 10000
) -> List[TextContent]:
    """
    Smart scroll to bring an element into view using natural language description
    This tool coordinates with Moondream agent for element detection

    Args:
        element_description: Natural language description of the element to scroll to
        alignment: Where to align the element ('top', 'center', 'bottom')
        timeout: Maximum time to wait for element detection in milliseconds
    """
    try:
        current_page = await ensure_browser()

        result = {
            'success': True,
            'tool': 'smart_scroll_to_element',
            'element_description': element_description,
            'alignment': alignment,
            'timeout': timeout,
            'coordination_trigger': True,
            'status': 'ready_for_coordinate_detection',
            'message': f'Smart scroll request prepared for element: "{element_description}"',
            'instructions_for_client': {
                'action': 'coordinate_detection_required',
                'interaction_type': 'scroll_to',
                'element_description': element_description,
                'scroll_parameters': {
                    'alignment': alignment
                },
                'requires_screenshot': True
            },
            'workflow_note': 'This response triggers dual model coordination - client will use Moondream for element detection'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Smart scroll preparation failed: {e}")
        error_result = {
            'success': False,
            'tool': 'smart_scroll_to_element',
            'error': str(e),
            'element_description': element_description,
            'message': f'Failed to prepare smart scroll for element: "{element_description}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def smart_hover_element(
    element_description: str,
    duration: int = 1000,
    timeout: int = 10000
) -> List[TextContent]:
    """
    Smart hover over an element using natural language description
    This tool coordinates with Moondream agent for precise coordinate detection

    Args:
        element_description: Natural language description of the element to hover over
        duration: How long to hover in milliseconds
        timeout: Maximum time to wait for element detection in milliseconds
    """
    try:
        current_page = await ensure_browser()

        result = {
            'success': True,
            'tool': 'smart_hover_element',
            'element_description': element_description,
            'duration': duration,
            'timeout': timeout,
            'coordination_trigger': True,
            'status': 'ready_for_coordinate_detection',
            'message': f'Smart hover request prepared for element: "{element_description}"',
            'instructions_for_client': {
                'action': 'coordinate_detection_required',
                'interaction_type': 'hover',
                'element_description': element_description,
                'hover_parameters': {
                    'duration': duration
                },
                'requires_screenshot': True
            },
            'workflow_note': 'This response triggers dual model coordination - client will use Moondream for coordinate detection'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Smart hover preparation failed: {e}")
        error_result = {
            'success': False,
            'tool': 'smart_hover_element',
            'error': str(e),
            'element_description': element_description,
            'message': f'Failed to prepare smart hover for element: "{element_description}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def smart_wait_for_element(
    element_description: str,
    action: str = 'visible',
    timeout: int = 30000
) -> List[TextContent]:
    """
    Smart wait for an element to appear, disappear, or become interactable
    This tool coordinates with Moondream agent for element detection

    Args:
        element_description: Natural language description of the element to wait for
        action: What to wait for ('visible', 'hidden', 'clickable', 'text_change')
        timeout: Maximum time to wait in milliseconds
    """
    try:
        current_page = await ensure_browser()

        result = {
            'success': True,
            'tool': 'smart_wait_for_element',
            'element_description': element_description,
            'action': action,
            'timeout': timeout,
            'coordination_trigger': True,
            'status': 'ready_for_element_monitoring',
            'message': f'Smart wait request prepared for element: "{element_description}" to become {action}',
            'instructions_for_client': {
                'action': 'element_monitoring_required',
                'interaction_type': 'wait',
                'element_description': element_description,
                'wait_parameters': {
                    'action': action,
                    'timeout': timeout
                },
                'requires_screenshot': True
            },
            'workflow_note': 'This response triggers dual model coordination - client will use Moondream for element monitoring'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Smart wait preparation failed: {e}")
        error_result = {
            'success': False,
            'tool': 'smart_wait_for_element',
            'error': str(e),
            'element_description': element_description,
            'action': action,
            'message': f'Failed to prepare smart wait for element: "{element_description}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def smart_extract_text(
    element_description: str,
    attribute: str = 'textContent',
    timeout: int = 10000
) -> List[TextContent]:
    """
    Smart text extraction from an element using natural language description
    This tool coordinates with Moondream agent for precise element detection

    Args:
        element_description: Natural language description of the element to extract text from
        attribute: What to extract ('textContent', 'innerText', 'value', 'href', 'src')
        timeout: Maximum time to wait for element detection in milliseconds
    """
    try:
        current_page = await ensure_browser()

        result = {
            'success': True,
            'tool': 'smart_extract_text',
            'element_description': element_description,
            'attribute': attribute,
            'timeout': timeout,
            'coordination_trigger': True,
            'status': 'ready_for_coordinate_detection',
            'message': f'Smart text extraction request prepared for element: "{element_description}"',
            'instructions_for_client': {
                'action': 'coordinate_detection_required',
                'interaction_type': 'extract',
                'element_description': element_description,
                'extract_parameters': {
                    'attribute': attribute
                },
                'requires_screenshot': True
            },
            'workflow_note': 'This response triggers dual model coordination - client will use Moondream for element detection'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Smart text extraction preparation failed: {e}")
        error_result = {
            'success': False,
            'tool': 'smart_extract_text',
            'error': str(e),
            'element_description': element_description,
            'attribute': attribute,
            'message': f'Failed to prepare smart text extraction for element: "{element_description}"'
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


if __name__ == "__main__":
    # Add startup logging
    logger.info("Starting Enhanced Playwright MCP Server for Dual Model System")
    logger.info("Server optimized for Core LLM + Moondream Agent coordination")
    logger.info("Available tools: screenshot, click, type, navigate, scroll, and more")

    # Run the server
    asyncio.run(mcp.run())