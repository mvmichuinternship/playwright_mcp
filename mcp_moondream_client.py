"""
Dual Model Browser Automation System with Enhanced Logging
Shows detailed coordination between Core LLM and Moondream Agent
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time

from system_prompts import BROWSER_AUTOMATION_PROMPT

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dual_model_coordination.log')
    ]
)
logger = logging.getLogger(__name__)

# Create specific loggers for coordination tracking
coordination_logger = logging.getLogger('dual_model.coordination')
moondream_logger = logging.getLogger('dual_model.moondream')
core_llm_logger = logging.getLogger('dual_model.core_llm')

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    api_url: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1000


moondream_logger = logging.getLogger('dual_model.moondream')
import aiohttp
from PIL import Image
import base64
import io
import time
import moondream

class MoondreamAgent:
    """Moondream agent using the Python library for coordinate detection"""

    def __init__(self, config):
        self.config = config
        # Import and initialize the Moondream model
        try:
            from moondream import vl
            self.model = vl(endpoint="http://localhost:2020/v1")
        except ImportError:
            raise ImportError("Moondream library not installed. Install with: pip install moondream")

    async def find_coordinates(self, element_description: str, screenshot_data: str, context: str = "") -> Tuple[Optional[Tuple[int, int]], str]:
        """Find coordinates using Moondream's point method"""

        start_time = time.time()

        moondream_logger.info("=" * 80)
        moondream_logger.info("🎯 MOONDREAM LIBRARY COORDINATE DETECTION")
        moondream_logger.info("=" * 80)
        moondream_logger.info(f"📍 Element to find: '{element_description}'")
        moondream_logger.info(f"🌐 Page context: {context}")
        moondream_logger.info(f"📸 Screenshot data size: {len(screenshot_data)} bytes")

        try:
            # Convert base64 screenshot to PIL Image
            image_bytes = base64.b64decode(screenshot_data)
            image = Image.open(io.BytesIO(image_bytes))

            moondream_logger.info(f"📷 Image dimensions: {image.size}")
            moondream_logger.info("🚀 Running Moondream point detection...")

            # Use the model.point method directly
            result = self.model.point(image, element_description)
            processing_time = time.time() - start_time

            moondream_logger.info("✅ MOONDREAM POINT METHOD RESPONSE:")
            moondream_logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
            moondream_logger.info("📤 Raw response:")
            moondream_logger.info("-" * 40)
            moondream_logger.info(str(result))
            moondream_logger.info("-" * 40)

            # Extract coordinates from the points
            coordinates = self._extract_coordinates_from_points(result, image.size)

            if coordinates:
                x, y = coordinates
                moondream_logger.info("🎯 COORDINATES EXTRACTED:")
                moondream_logger.info(f"📍 Found coordinates: ({x}, {y})")
                moondream_logger.info(f"✅ Element '{element_description}' located successfully")
            else:
                moondream_logger.warning("❌ NO COORDINATES FOUND:")
                moondream_logger.warning(f"🔍 Could not find element: '{element_description}'")

            moondream_logger.info("=" * 80)
            moondream_logger.info("🏁 MOONDREAM COORDINATE DETECTION COMPLETE")
            moondream_logger.info("=" * 80)

            return coordinates, str(result)

        except Exception as e:
            processing_time = time.time() - start_time
            moondream_logger.error("💥 MOONDREAM LIBRARY ERROR:")
            moondream_logger.error(f"⏱️ Failed after: {processing_time:.2f} seconds")
            moondream_logger.error(f"❌ Error: {str(e)}")
            return None, f"Error: {str(e)}"

    def _extract_coordinates_from_points(self, result: dict, image_size: tuple) -> Optional[Tuple[int, int]]:
        """Extract coordinates from Moondream point method response"""

        moondream_logger.info("🔍 EXTRACTING COORDINATES FROM POINTS:")

        try:
            # Get points from the result
            points = result.get('points', [])

            if not points:
                moondream_logger.warning("❌ No points found in response")
                return None

            # Take the first point (most confident detection)
            first_point = points[0]
            moondream_logger.info(f"📍 First point data: {first_point}")

            # Extract normalized coordinates and convert to absolute pixels
            w, h = image_size
            x_normalized = first_point['x']
            y_normalized = first_point['y']

            # Convert normalized coordinates (0-1) to absolute pixel coordinates
            x = int(x_normalized * w)
            y = int(y_normalized * h)

            moondream_logger.info(f"✅ Converted coordinates: normalized ({x_normalized:.3f}, {y_normalized:.3f}) -> absolute ({x}, {y})")
            moondream_logger.info(f"📐 Image size: {w}x{h}")

            return (x, y)

        except (KeyError, IndexError, TypeError, ValueError) as e:
            moondream_logger.error(f"⚠️ Error extracting coordinates: {e}")
            return None

    async def analyze_page(self, screenshot_data: str, question: str = "Describe this webpage") -> str:
        """Analyze page content using Moondream library"""

        start_time = time.time()

        moondream_logger.info("=" * 80)
        moondream_logger.info("🔍 MOONDREAM PAGE ANALYSIS REQUEST")
        moondream_logger.info("=" * 80)
        moondream_logger.info(f"❓ Analysis question: '{question}'")
        moondream_logger.info(f"📸 Screenshot data size: {len(screenshot_data)} bytes")

        try:
            # Convert base64 screenshot to PIL Image
            image_bytes = base64.b64decode(screenshot_data)
            image = Image.open(io.BytesIO(image_bytes))

            moondream_logger.info("🚀 Running Moondream query...")

            # Use the model.query method
            result = self.model.query(image, question)
            processing_time = time.time() - start_time

            moondream_logger.info("✅ MOONDREAM PAGE ANALYSIS COMPLETE:")
            moondream_logger.info(f"⏱️ Analysis time: {processing_time:.2f} seconds")

            answer = result.get('answer', str(result))
            moondream_logger.info("📄 Analysis result:")
            moondream_logger.info("-" * 40)
            moondream_logger.info(answer[:500] + "..." if len(answer) > 500 else answer)
            moondream_logger.info("-" * 40)
            moondream_logger.info("=" * 80)

            return answer

        except Exception as e:
            processing_time = time.time() - start_time
            moondream_logger.error(f"💥 Moondream page analysis failed after {processing_time:.2f}s: {e}")
            return f"Error analyzing page: {str(e)}"

    def get_caption(self, screenshot_data: str) -> str:
        """Get image caption using Moondream library"""

        try:
            # Convert base64 screenshot to PIL Image
            image_bytes = base64.b64decode(screenshot_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Use the model.caption method
            result = self.model.caption(image)
            return result.get('caption', '')

        except Exception as e:
            moondream_logger.error(f"💥 Moondream captioning failed: {e}")
            return f"Error generating caption: {str(e)}"

    def detect_objects(self, screenshot_data: str, object_name: str) -> list:
        """Detect objects using Moondream library"""

        try:
            # Convert base64 screenshot to PIL Image
            image_bytes = base64.b64decode(screenshot_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Use the model.detect method
            result = self.model.detect(image, object_name)
            return result.get('objects', [])

        except Exception as e:
            moondream_logger.error(f"💥 Moondream detection failed: {e}")
            return []

# Example configuration for the updated agent
def create_moondream_config():
    """Create configuration for Moondream agent using point method"""
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        api_url: str
        model_name: str
        temperature: float = 0.0
        max_tokens: int = 300

    return ModelConfig(
        api_url="http://localhost:2020/v1",  # Direct to Moondream API, no /v1
        model_name="moondream",
        temperature=0.0,
        max_tokens=300
    )

class MCPToolInput(BaseModel):
    """Dynamic input schema for MCP tools"""
    pass

class EnhancedMCPTool(BaseTool):
    """Enhanced MCP tool with dual model coordination logging"""
    mcp_client: 'DualModelMCPClient' = Field(exclude=True)
    def __init__(self, tool_name: str, tool_description: str, mcp_client: 'DualModelMCPClient', input_schema: Optional[Dict] = None):
        # Create dynamic input model if schema is provided
        args_schema = None

        if input_schema and input_schema.get("properties"):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            annotations = {}
            field_info = {}

            for prop_name, prop_info in properties.items():
                if prop_info.get("type") == "integer":
                    field_type = int
                elif prop_info.get("type") == "boolean":
                    field_type = bool
                elif prop_info.get("type") == "array":
                    field_type = List[str]
                else:
                    field_type = str

                if prop_name in required:
                    annotations[prop_name] = field_type
                    field_info[prop_name] = Field(description=prop_info.get("description", ""))
                else:
                    annotations[prop_name] = Optional[field_type]
                    field_info[prop_name] = Field(default=None, description=prop_info.get("description", ""))

            if annotations:
                DynamicInput = type(
                    f"{tool_name}Input",
                    (BaseModel,),
                    {
                        "__annotations__": annotations,
                        **field_info
                    }
                )
                args_schema = DynamicInput

        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema
        )
        self.mcp_client = mcp_client

    def _run(self, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Enhanced async tool execution with coordination logging"""
        try:
            core_llm_logger.info("="*60)
            core_llm_logger.info(f"🔧 CORE LLM INVOKING TOOL: {self.name}")
            core_llm_logger.info("="*60)
            core_llm_logger.info(f"📥 Tool parameters: {kwargs}")

            # Filter out None values
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            # Enforce screenshot-first workflow for interaction tools
            ui_interaction_tools = ['smart_click_element', 'smart_input_text', 'click_at_coordinates']

            if self.name in ui_interaction_tools:
                core_llm_logger.info(f"🎯 UI interaction tool detected: {self.name}")
                if not self.mcp_client.has_recent_screenshot():
                    core_llm_logger.info("📸 No recent screenshot available - taking new screenshot")
                    await self.mcp_client.call_tool('take_screenshot_and_store', {})
                else:
                    core_llm_logger.info("✅ Recent screenshot available - proceeding with interaction")

            start_time = time.time()
            result = await self.mcp_client.call_tool(self.name, filtered_kwargs)
            execution_time = time.time() - start_time

            core_llm_logger.info(f"✅ Tool '{self.name}' completed in {execution_time:.2f}s")
            core_llm_logger.info("="*60)

            return await self._process_mcp_response(result)

        except Exception as e:
            core_llm_logger.error(f"💥 Error calling {self.name}: {e}")
            return f"Error calling {self.name}: {str(e)}"

    async def _process_mcp_response(self, result: Any) -> str:
        """Process MCP response with enhanced feedback"""
        if isinstance(result, dict) and "content" in result:
            return await self._process_content_items(result["content"])

        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    async def _process_content_items(self, content_items: List[Dict]) -> str:
        """Process content items from MCP response"""
        if not isinstance(content_items, list):
            return str(content_items)

        text_parts = []
        image_count = 0

        for item in content_items:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                image_count += 1
                self.mcp_client.mark_screenshot_available()
                text_parts.append(f"📸 Screenshot captured for Moondream analysis")

        result_text = "\n".join(text_parts) if text_parts else "Operation completed"

        if image_count > 0:
            result_text += f"\n\n✅ {image_count} screenshot(s) ready for coordinate detection"

        return result_text

class DualModelMCPClient:
    """Enhanced MCP Client with dual model architecture and detailed coordination logging"""

    def __init__(self, server_command: List[str], moondream_agent: MoondreamAgent):
        self.server_command = server_command
        self.moondream_agent = moondream_agent
        self.process = None
        self.tools_info = {}
        self.request_id = 0
        self.screenshot_timestamp = 0
        self.screenshot_available = False
        self.last_screenshot_data = None

    def has_recent_screenshot(self) -> bool:
        """Check if we have a recent screenshot (within last 30 seconds)"""
        import time
        is_recent = self.screenshot_available and (time.time() - self.screenshot_timestamp) < 30
        coordination_logger.debug(f"📸 Screenshot status: available={self.screenshot_available}, recent={is_recent}")
        return is_recent

    def mark_screenshot_available(self, screenshot_data: str = None):
        """Mark that a screenshot is now available"""
        import time
        self.screenshot_available = True
        self.screenshot_timestamp = time.time()
        if screenshot_data:
            self.last_screenshot_data = screenshot_data
        coordination_logger.info("📸 Screenshot marked as available for Moondream analysis")

    def mark_screenshot_stale(self):
        """Mark screenshot as stale (after navigation, scrolling, etc.)"""
        self.screenshot_available = False
        self.last_screenshot_data = None
        coordination_logger.info("🔄 Screenshot marked as stale - will require new capture")

    async def start(self):
        """Start the MCP server process"""
        try:
            coordination_logger.info("🚀 Starting MCP server for dual model coordination...")
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024*100  # 100MB buffer for large images
            )

            await asyncio.sleep(3)  # Give server time to initialize

            # Initialize MCP protocol
            init_response = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "dual-model-browser-automation",
                    "version": "1.0.0"
                }
            })

            logger.info(f"MCP Server initialized: {init_response}")
            await self._send_notification("notifications/initialized")

            # List available tools
            tools_response = await self._send_request("tools/list", {})
            if "tools" in tools_response:
                for tool in tools_response["tools"]:
                    self.tools_info[tool["name"]] = tool

            coordination_logger.info(f"✅ MCP server connected. Available tools: {list(self.tools_info.keys())}")

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    def _get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send request to MCP server"""
        if not self.process:
            raise Exception("MCP server not started")

        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params or {}
        }

        request_json = json.dumps(request) + "\n"
        logger.debug(f"Sending request: {method}")

        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()

        # Enhanced response reading
        response_buffer = b""
        timeout = 120  # 2 minutes timeout
        chunk_size = 65536

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                chunk = await asyncio.wait_for(
                    self.process.stdout.read(chunk_size),
                    timeout=10
                )

                if not chunk:
                    await asyncio.sleep(0.1)
                    continue

                response_buffer += chunk

                # Try to parse complete responses
                try:
                    response_text = response_buffer.decode('utf-8', errors='ignore')
                    lines = response_text.split('\n')

                    for line in lines[:-1]:
                        line = line.strip()
                        if line:
                            try:
                                response = json.loads(line)
                                if "error" in response:
                                    raise Exception(f"MCP Error: {response['error']}")
                                if "result" in response and response.get("id"):
                                    return response["result"]
                            except json.JSONDecodeError:
                                continue

                    response_buffer = lines[-1].encode('utf-8')

                except UnicodeDecodeError:
                    continue

            except asyncio.TimeoutError:
                continue

        raise Exception(f"MCP server response timeout after {timeout} seconds")

    async def _send_notification(self, method: str, params: Optional[Dict] = None):
        """Send notification to MCP server"""
        if not self.process:
            raise Exception("MCP server not started")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }

        notification_json = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()

    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Call a tool on the MCP server with dual model coordination and detailed logging"""

        coordination_logger.info("🎯 DUAL MODEL COORDINATION TRIGGERED")
        coordination_logger.info(f"🔧 Tool requested: {tool_name}")
        coordination_logger.info(f"📥 Arguments: {arguments}")

        # Mark screenshot as stale for navigation/scrolling actions
        if tool_name in ['navigate', 'scroll_page', 'press_key']:
            coordination_logger.info(f"🔄 Navigation/action tool '{tool_name}' - marking screenshot stale")
            self.mark_screenshot_stale()

        # For smart interaction tools, use Moondream agent for coordinate detection
        if tool_name == 'smart_click_element' and 'element_description' in arguments:
            coordination_logger.info("🎯 SMART CLICK COORDINATION INITIATED")
            coordination_logger.info("📍 Core LLM → Moondream Agent coordinate detection handoff")
            return await self._handle_smart_click(arguments)

        elif tool_name == 'smart_input_text' and 'element_description' in arguments:
            coordination_logger.info("📝 SMART INPUT COORDINATION INITIATED")
            coordination_logger.info("📍 Core LLM → Moondream Agent coordinate detection handoff")
            return await self._handle_smart_input(arguments)

        elif tool_name == 'analyze_page_content':
            coordination_logger.info("🔍 PAGE ANALYSIS COORDINATION INITIATED")
            coordination_logger.info("📸 Core LLM → Moondream Agent page analysis handoff")
            return await self._handle_page_analysis(arguments)

        # Standard MCP tool call
        coordination_logger.info(f"🔧 Standard MCP tool call: {tool_name}")
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        # Mark screenshot as available for screenshot tools
        if tool_name == 'take_screenshot_and_store':
            coordination_logger.info("📸 Screenshot tool completed - extracting image data")
            self.mark_screenshot_available()
            # Extract screenshot data from result if available
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if item.get("type") == "image" and "data" in item:
                        self.last_screenshot_data = item["data"]
                        coordination_logger.info(f"✅ Screenshot data extracted: {len(self.last_screenshot_data)} bytes")

        return result

    async def _handle_smart_click(self, arguments: Dict) -> Dict:
        """Handle smart click using Moondream agent with detailed coordination logging"""
        element_description = arguments.get('element_description')
        button = arguments.get('button', 'left')
        delay = arguments.get('delay', 100)

        coordination_logger.info("🎯 SMART CLICK COORDINATION SEQUENCE")
        coordination_logger.info("="*80)
        coordination_logger.info("📋 COORDINATION PLAN:")
        coordination_logger.info("1. 🔍 Core LLM identifies need for element click")
        coordination_logger.info("2. 📸 Ensure fresh screenshot is available")
        coordination_logger.info("3. 🌐 Gather page context information")
        coordination_logger.info("4. 📍 Moondream Agent: Find element coordinates")
        coordination_logger.info("5. 🖱️ Execute click at detected coordinates")
        coordination_logger.info("6. 📸 Capture post-click screenshot")
        coordination_logger.info("="*80)

        try:
            # Step 1: Ensure we have a screenshot
            coordination_logger.info("📸 STEP 1: Ensuring screenshot availability")
            if not self.last_screenshot_data:
                coordination_logger.info("📸 No screenshot data available - capturing new screenshot")
                await self.call_tool('take_screenshot_and_store', {})
                coordination_logger.info("✅ Fresh screenshot captured and ready for Moondream")
            else:
                coordination_logger.info(f"✅ Screenshot data available: {len(self.last_screenshot_data)} bytes")

            # Step 2: Get page context
            coordination_logger.info("🌐 STEP 2: Gathering page context for Moondream")
            page_info = await self._send_request("tools/call", {
                "name": "get_page_info",
                "arguments": {}
            })

            context = ""
            if isinstance(page_info, dict) and "content" in page_info:
                for item in page_info["content"]:
                    if item.get("type") == "text":
                        try:
                            info_data = json.loads(item.get("text", "{}"))
                            context = f"Page: {info_data.get('title', '')} - {info_data.get('url', '')}"
                            coordination_logger.info(f"✅ Page context extracted: {context}")
                        except:
                            coordination_logger.warning("⚠️ Could not parse page context")

            # Step 3: CORE LLM → MOONDREAM HANDOFF
            coordination_logger.info("🤝 STEP 3: CORE LLM → MOONDREAM COORDINATION HANDOFF")
            coordination_logger.info("📤 Sending to Moondream:")
            coordination_logger.info(f"  - Element: '{element_description}'")
            coordination_logger.info(f"  - Context: '{context}'")
            coordination_logger.info(f"  - Screenshot: {len(self.last_screenshot_data)} bytes")

            # Use Moondream agent to find coordinates
            coordinates, moondream_response = await self.moondream_agent.find_coordinates(
                element_description, self.last_screenshot_data, context
            )

            coordination_logger.info("📥 MOONDREAM RESPONSE RECEIVED")
            if not coordinates:
                coordination_logger.error("❌ COORDINATION FAILURE: Moondream could not find coordinates")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "element_description": element_description,
                                "error": "Moondream could not find element coordinates",
                                "moondream_response": moondream_response,
                                "coordination_status": "FAILED - Coordinate detection unsuccessful"
                            }, indent=2)
                        }
                    ]
                }

            x, y = coordinates
            coordination_logger.info("✅ COORDINATION SUCCESS: Coordinates received from Moondream")
            coordination_logger.info(f"📍 Target coordinates: ({x}, {y})")

            # Step 4: Execute click
            coordination_logger.info("🖱️ STEP 4: Executing click at Moondream-detected coordinates")
            click_result = await self._send_request("tools/call", {
                "name": "click_at_coordinates",
                "arguments": {"x": x, "y": y, "button": button, "delay": delay}
            })

            coordination_logger.info(f"✅ Click executed at ({x}, {y})")

            # Step 5: Post-click screenshot
            coordination_logger.info("📸 STEP 5: Capturing post-click screenshot")
            await asyncio.sleep(1)
            await self.call_tool('take_screenshot_and_store', {})

            coordination_logger.info("🎉 SMART CLICK COORDINATION COMPLETE")
            coordination_logger.info("="*80)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "element_description": element_description,
                            "coordinates": {"x": x, "y": y},
                            "button": button,
                            "delay": delay,
                            "moondream_response": moondream_response,
                            "coordination_status": "SUCCESS - Core LLM + Moondream coordination complete",
                            "message": f'Successfully clicked "{element_description}" at coordinates ({x}, {y}) using Moondream coordinate detection'
                        }, indent=2)
                    }
                ]
            }

        except Exception as e:
            coordination_logger.error(f"💥 SMART CLICK COORDINATION FAILED: {str(e)}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": False,
                            "element_description": element_description,
                            "error": str(e),
                            "coordination_status": "FAILED - Exception during coordination"
                        }, indent=2)
                    }
                ]
            }

    async def _handle_smart_input(self, arguments: Dict) -> Dict:
        """Handle smart input using Moondream agent with detailed coordination logging"""
        element_description = arguments.get('element_description')
        text = arguments.get('text')
        clear_first = arguments.get('clear_first', True)
        delay = arguments.get('delay', 50)

        coordination_logger.info("📝 SMART INPUT COORDINATION SEQUENCE")
        coordination_logger.info("="*80)
        coordination_logger.info("📋 COORDINATION PLAN:")
        coordination_logger.info("1. 🔍 Core LLM identifies need for text input")
        coordination_logger.info("2. 📸 Ensure fresh screenshot is available")
        coordination_logger.info("3. 🌐 Gather page context information")
        coordination_logger.info("4. 📍 Moondream Agent: Find input field coordinates")
        coordination_logger.info("5. 🖱️ Click on input field")
        coordination_logger.info("6. ⌨️ Execute text input")
        coordination_logger.info("7. 📸 Capture post-input screenshot")
        coordination_logger.info("="*80)

        try:
            # Step 1: Ensure we have a screenshot
            coordination_logger.info("📸 STEP 1: Ensuring screenshot availability for input field detection")
            if not self.last_screenshot_data:
                coordination_logger.info("📸 No screenshot data available - capturing new screenshot")
                await self.call_tool('take_screenshot_and_store', {})
            else:
                coordination_logger.info(f"✅ Screenshot data available: {len(self.last_screenshot_data)} bytes")

            # Step 2: Get page context
            coordination_logger.info("🌐 STEP 2: Gathering page context")
            page_info = await self._send_request("tools/call", {
                "name": "get_page_info",
                "arguments": {}
            })

            context = ""
            if isinstance(page_info, dict) and "content" in page_info:
                for item in page_info["content"]:
                    if item.get("type") == "text":
                        try:
                            info_data = json.loads(item.get("text", "{}"))
                            context = f"Page: {info_data.get('title', '')} - {info_data.get('url', '')}"
                            coordination_logger.info(f"✅ Page context: {context}")
                        except:
                            coordination_logger.warning("⚠️ Could not parse page context")

            # Step 3: CORE LLM → MOONDREAM HANDOFF FOR INPUT FIELD
            coordination_logger.info("🤝 STEP 3: CORE LLM → MOONDREAM INPUT FIELD DETECTION")
            field_description = f"input field: {element_description}"
            coordination_logger.info("📤 Sending to Moondream:")
            coordination_logger.info(f"  - Input field: '{field_description}'")
            coordination_logger.info(f"  - Text to input: '{text}'")
            coordination_logger.info(f"  - Context: '{context}'")

            coordinates, moondream_response = await self.moondream_agent.find_coordinates(
                field_description, self.last_screenshot_data, context
            )

            if not coordinates:
                coordination_logger.error("❌ COORDINATION FAILURE: Moondream could not find input field")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "element_description": element_description,
                                "text": text,
                                "error": "Moondream could not find input field coordinates",
                                "moondream_response": moondream_response,
                                "coordination_status": "FAILED - Input field detection unsuccessful"
                            }, indent=2)
                        }
                    ]
                }

            x, y = coordinates
            coordination_logger.info("✅ COORDINATION SUCCESS: Input field coordinates received")
            coordination_logger.info(f"📍 Input field coordinates: ({x}, {y})")

            # Step 4: Click on input field
            coordination_logger.info("🖱️ STEP 4: Clicking input field at Moondream coordinates")
            await self._send_request("tools/call", {
                "name": "click_at_coordinates",
                "arguments": {"x": x, "y": y, "delay": 100}
            })

            await asyncio.sleep(0.5)

            # Step 5: Clear field if requested
            if clear_first:
                coordination_logger.info("🗑️ STEP 5: Clearing input field")
                await self._send_request("tools/call", {
                    "name": "press_key",
                    "arguments": {"key": "Control+a"}
                })
                await self._send_request("tools/call", {
                    "name": "press_key",
                    "arguments": {"key": "Delete"}
                })
                await asyncio.sleep(0.2)

            # Step 6: Type the text
            coordination_logger.info(f"⌨️ STEP 6: Typing text: '{text}'")
            if delay > 0:
                coordination_logger.info(f"⌨️ Using character-by-character input with {delay}ms delay")
                for i, char in enumerate(text):
                    await self._send_request("tools/call", {
                        "name": "press_key",
                        "arguments": {"key": char}
                    })
                    await asyncio.sleep(delay / 1000)
                    if i % 10 == 9:  # Log progress every 10 characters
                        coordination_logger.debug(f"⌨️ Typed {i+1}/{len(text)} characters")
            else:
                coordination_logger.info("⌨️ Using direct text input")
                for char in text:
                    await self._send_request("tools/call", {
                        "name": "press_key",
                        "arguments": {"key": char}
                    })

            # Step 7: Post-input screenshot
            coordination_logger.info("📸 STEP 7: Capturing post-input screenshot")
            await asyncio.sleep(0.5)
            await self.call_tool('take_screenshot_and_store', {})

            coordination_logger.info("🎉 SMART INPUT COORDINATION COMPLETE")
            coordination_logger.info("="*80)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "element_description": element_description,
                            "coordinates": {"x": x, "y": y},
                            "text": text,
                            "cleared_first": clear_first,
                            "moondream_response": moondream_response,
                            "coordination_status": "SUCCESS - Core LLM + Moondream input coordination complete",
                            "message": f'Successfully typed "{text}" into "{element_description}" at coordinates ({x}, {y}) using Moondream coordinate detection'
                        }, indent=2)
                    }
                ]
            }

        except Exception as e:
            coordination_logger.error(f"💥 SMART INPUT COORDINATION FAILED: {str(e)}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": False,
                            "element_description": element_description,
                            "text": text,
                            "error": str(e),
                            "coordination_status": "FAILED - Exception during input coordination"
                        }, indent=2)
                    }
                ]
            }

    async def _handle_page_analysis(self, arguments: Dict) -> Dict:
        """Handle page analysis using Moondream agent with coordination logging"""
        question = arguments.get('question', 'Describe this webpage and list all interactive elements')

        coordination_logger.info("🔍 PAGE ANALYSIS COORDINATION SEQUENCE")
        coordination_logger.info("="*80)
        coordination_logger.info("📋 COORDINATION PLAN:")
        coordination_logger.info("1. 🔍 Core LLM requests page analysis")
        coordination_logger.info("2. 📸 Ensure fresh screenshot is available")
        coordination_logger.info("3. 📍 Moondream Agent: Analyze page content")
        coordination_logger.info("4. 📤 Return analysis to Core LLM")
        coordination_logger.info("="*80)

        try:
            # Step 1: Ensure we have a screenshot
            coordination_logger.info("📸 STEP 1: Ensuring screenshot for page analysis")
            if not self.last_screenshot_data:
                coordination_logger.info("📸 No screenshot data - capturing for analysis")
                await self.call_tool('take_screenshot_and_store', {})
            else:
                coordination_logger.info(f"✅ Screenshot available: {len(self.last_screenshot_data)} bytes")

            # Step 2: CORE LLM → MOONDREAM HANDOFF FOR PAGE ANALYSIS
            coordination_logger.info("🤝 STEP 2: CORE LLM → MOONDREAM PAGE ANALYSIS HANDOFF")
            coordination_logger.info("📤 Sending to Moondream:")
            coordination_logger.info(f"  - Analysis question: '{question}'")
            coordination_logger.info(f"  - Screenshot size: {len(self.last_screenshot_data)} bytes")

            # Use Moondream agent to analyze the page
            analysis = await self.moondream_agent.analyze_page(self.last_screenshot_data, question)

            coordination_logger.info("✅ COORDINATION SUCCESS: Page analysis received from Moondream")
            coordination_logger.info("🎉 PAGE ANALYSIS COORDINATION COMPLETE")
            coordination_logger.info("="*80)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "question": question,
                            "analysis": analysis,
                            "coordination_status": "SUCCESS - Core LLM + Moondream analysis coordination complete",
                            "message": "Page analysis completed using Moondream vision model"
                        }, indent=2)
                    }
                ]
            }

        except Exception as e:
            coordination_logger.error(f"💥 PAGE ANALYSIS COORDINATION FAILED: {str(e)}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": False,
                            "question": question,
                            "error": str(e),
                            "coordination_status": "FAILED - Exception during analysis coordination"
                        }, indent=2)
                    }
                ]
            }

    def create_langchain_tools(self) -> List[EnhancedMCPTool]:
        """Create LangChain tools with proper prioritization"""
        langchain_tools = []

        # Core workflow tools (highest priority)
        core_tools = [
            'take_screenshot_and_store',
            'smart_click_element',    # Now uses Moondream agent
            'smart_input_text',       # Now uses Moondream agent
            'analyze_page_content'    # Now uses Moondream agent
        ]

        # Essential browser tools
        browser_tools = [
            'launch_browser',
            'navigate',
            'get_page_info',
            'scroll_page',
            'press_key',
            'close_browser'
        ]

        # Coordinate-based tools (for fallback)
        coordinate_tools = [
            'click_at_coordinates'
        ]

        all_tools = core_tools + browser_tools + coordinate_tools

        coordination_logger.info(f"🔧 Creating LangChain tools for dual model coordination")

        for tool_name in all_tools:
            if tool_name in self.tools_info:
                try:
                    tool = EnhancedMCPTool(
                        tool_name=tool_name,
                        tool_description=self.tools_info[tool_name].get("description", f"MCP tool: {tool_name}"),
                        mcp_client=self,
                        input_schema=self.tools_info[tool_name].get("inputSchema")
                    )
                    langchain_tools.append(tool)
                    coordination_logger.debug(f"✅ Created tool: {tool_name}")
                except Exception as e:
                    coordination_logger.warning(f"⚠️ Failed to create tool {tool_name}: {e}")

        coordination_logger.info(f"✅ Created {len(langchain_tools)} tools for Core LLM")
        return langchain_tools

class DualModelBrowserAutomationAgent:
    """Browser automation agent with separate Core LLM and Moondream agents and detailed logging"""

    def __init__(self, core_llm_config: ModelConfig, moondream_config: ModelConfig):
        self.core_llm_config = core_llm_config
        self.moondream_agent = MoondreamAgent(moondream_config)
        self.mcp_client = None
        self.agent_executor = None

    async def setup(self, mcp_server_script_path: str):
        """Setup agent with dual model architecture and detailed logging"""
        try:
            logger.info("🚀 DUAL MODEL BROWSER AUTOMATION AGENT SETUP")
            logger.info("="*80)

            # Initialize dual model MCP client
            logger.info("🔧 Initializing dual model MCP client...")
            self.mcp_client = DualModelMCPClient(
                server_command=["python", mcp_server_script_path],
                moondream_agent=self.moondream_agent
            )
            await self.mcp_client.start()

            # Create tools
            logger.info("🛠️ Creating coordination tools...")
            tools = self.mcp_client.create_langchain_tools()
            if not tools:
                raise ValueError("No tools were successfully created from MCP server")

            logger.info(f"✅ Created {len(tools)} tools from MCP server")

            # Setup core LLM
            logger.info("🤖 Setting up Core LLM for orchestration...")
            core_llm = ChatOpenAI(
                base_url=self.core_llm_config.api_url,
                api_key="not-needed",
                model=self.core_llm_config.model_name,
                temperature=self.core_llm_config.temperature,
                max_tokens=self.core_llm_config.max_tokens
            )

            # Test core LLM connection
            try:
                logger.info("🔍 Testing Core LLM connection...")
                test_response = await core_llm.ainvoke("Respond with 'Core LLM ready for browser automation orchestration'")
                core_llm_logger.info(f"✅ Core LLM connection test successful: {test_response.content}")
            except Exception as e:
                core_llm_logger.error(f"❌ Core LLM connection failed: {e}")
                raise ValueError(f"Cannot connect to Core LLM: {str(e)}")

            # Test Moondream connection
            try:
                logger.info("🌙 Testing Moondream agent connection...")
                test_analysis = await self.moondream_agent.analyze_page("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "Test connection")
                moondream_logger.info(f"✅ Moondream agent connection test successful")
            except Exception as e:
                moondream_logger.error(f"❌ Moondream agent connection failed: {e}")
                raise ValueError(f"Cannot connect to Moondream agent: {str(e)}")

            # Enhanced system prompt for dual model coordination
            system_prompt = BROWSER_AUTOMATION_PROMPT

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            # Create agent
            logger.info("🤝 Creating dual model agent executor...")
            agent = create_tool_calling_agent(core_llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=25,  # More iterations for complex workflows
                return_intermediate_steps=True
            )

            logger.info("✅ DUAL MODEL BROWSER AUTOMATION AGENT SETUP COMPLETE")
            logger.info(f"🤖 Core LLM: {self.core_llm_config.model_name} @ {self.core_llm_config.api_url}")
            logger.info(f"🌙 Moondream Agent: {self.moondream_agent.config.model_name} @ {self.moondream_agent.config.api_url}")
            logger.info("🎯 Coordination logging enabled - check dual_model_coordination.log for details")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"💥 Setup failed: {e}")
            if self.mcp_client:
                await self.mcp_client.stop()
            raise

    async def run(self, query: str) -> str:
        """Run the agent with dual model coordination and detailed logging"""
        if not self.agent_executor:
            raise ValueError("Agent not setup. Call setup() first.")

        try:
            start_time = time.time()

            logger.info("🎯 DUAL MODEL QUERY PROCESSING INITIATED")
            logger.info("="*80)
            logger.info(f"📝 User query: {query}")
            logger.info(f"🤖 Core LLM orchestrating: {self.core_llm_config.model_name}")
            logger.info(f"🌙 Moondream available for: coordinate detection, page analysis")
            logger.info("="*80)

            result = await self.agent_executor.ainvoke({"input": query})

            processing_time = time.time() - start_time
            output = result["output"]

            # Add coordination summary
            if "intermediate_steps" in result:
                steps = len(result["intermediate_steps"])

                logger.info("📊 DUAL MODEL COORDINATION SUMMARY")
                logger.info("="*60)
                logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds")
                logger.info(f"🔧 Core LLM orchestration steps: {steps}")
                logger.info("🤝 Coordination events logged to: dual_model_coordination.log")
                logger.info("="*60)

                output += f"\n\n🤖 Dual Model Coordination Summary:"
                output += f"\n- Core LLM orchestrated {steps} steps in {processing_time:.2f}s"
                output += f"\n- Moondream agent handled all coordinate detection"
                output += f"\n- Screenshot-first workflow enforced"
                output += f"\n- Detailed logs available in dual_model_coordination.log"

            return output

        except Exception as e:
            logger.error(f"💥 Error running dual model agent: {e}")
            return f"❌ Error: {str(e)}\n\nPlease ensure:\n- Core LLM is running and accessible\n- Moondream model is running and accessible\n- MCP server is available\n- Browser can be launched"

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up dual model agent...")
        if self.mcp_client:
            await self.mcp_client.stop()
        logger.info("✅ Cleanup complete")

async def main():
    """Main function with dual model configuration and enhanced logging"""
    import sys
    import os

    print("🚀 DUAL MODEL BROWSER AUTOMATION AGENT WITH ENHANCED LOGGING")
    print("="*80)
    print("📊 LOGGING FEATURES:")
    print("- 🤖 Core LLM orchestration logs")
    print("- 🌙 Moondream coordinate detection logs")
    print("- 🤝 Inter-model coordination tracking")
    print("- 📸 Screenshot workflow monitoring")
    print("- 📁 Detailed logs saved to: dual_model_coordination.log")
    print("="*80)

    # Configuration
    mcp_server_path = "enhanced_playwright_server.py"

    # Core LLM Configuration (for orchestration)
    core_llm_config = ModelConfig(
        api_url="http://localhost:1234/v1",
        model_name="qwen/qwen2.5-vl-7b",
        temperature=0.1,
        max_tokens=2000
    )

    # Moondream Configuration (for coordinate detection)
    moondream_config = ModelConfig(
        api_url="http://localhost:2020/v1",
        model_name="moondream",
        temperature=0.0,
        max_tokens=300
    )

    # Check if server file exists
    if not os.path.exists(mcp_server_path):
        print(f"❌ MCP server script not found: {mcp_server_path}")
        print("Please ensure the enhanced server script is in the current directory")
        return

    # Initialize the dual model agent
    agent = DualModelBrowserAutomationAgent(core_llm_config, moondream_config)

    try:
        print("\n🚀 Setting up Dual Model Browser Automation Agent...")
        print("🔄 This may take a moment to initialize both models...")

        await agent.setup(mcp_server_path)

        print("\n✅ Dual Model Browser Automation Agent is ready!")
        print("\n🤖 DUAL MODEL ARCHITECTURE:")
        print(f"- Core LLM (Orchestration): {core_llm_config.model_name}")
        print(f"- Moondream Agent (Vision): {moondream_config.model_name}")
        print("- Automatic screenshot-first workflow")
        print("- Precise coordinate detection via Moondream")
        print("- Smart element interaction based on AI vision")

        print("\n📊 ENHANCED LOGGING ENABLED:")
        print("- Real-time coordination tracking")
        print("- Moondream request/response logging")
        print("- Core LLM tool invocation logs")
        print("- Screenshot workflow monitoring")
        print("- Log file: dual_model_coordination.log")

        print("\n🎯 EXAMPLE COMMANDS:")
        print("- 'Launch browser and go to google.com'")
        print("- 'Click the search box and type hello world'")
        print("- 'Find and click the Google Search button'")
        print("- 'Analyze what is on this page'")
        print("- 'Fill out the login form with username test@example.com'")

        print("\n📝 SPECIAL COMMANDS:")
        print("- 'screenshot' - Take and analyze current page")
        print("- 'analyze' - Detailed page content analysis")
        print("- 'help' - Show available commands")

        print(f"\n🔧 Configuration:")
        print(f"- Core LLM: {core_llm_config.api_url} ({core_llm_config.model_name})")
        print(f"- Moondream: {moondream_config.api_url} ({moondream_config.model_name})")
        print(f"- MCP Server: {mcp_server_path}")
        print(f"- Dual Model Coordination: ✅ Enabled with Enhanced Logging")

        print("\n" + "="*70)
        print("Type 'quit' to exit | Watch dual_model_coordination.log for detailed logs")
        print("="*70)

        # Interactive loop
        while True:
            try:
                user_input = input("\n🎯 What would you like me to do? ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['help', 'h']:
                    print("\n📚 DUAL MODEL SYSTEM COMMANDS:")
                    print("Browser Control: launch, navigate, scroll, close")
                    print("Smart Interaction: click [element], type [text] in [field]")
                    print("Analysis: screenshot, analyze, describe page")
                    print("Examples:")
                    print("  'click login button' -> Core LLM + Moondream coordination")
                    print("  'type username in email field' -> Automatic coordinate detection")
                    print("  'analyze this page' -> Moondream visual analysis")
                    print("\n📊 Check dual_model_coordination.log for detailed coordination logs")
                    continue

                if user_input.lower() == 'screenshot':
                    user_input = "Take a screenshot of the current page"

                if user_input.lower() == 'analyze':
                    user_input = "Take a screenshot and analyze the page content in detail"

                print(f"\n🤖 Processing: {user_input}")
                print("🔄 Core LLM orchestrating with Moondream vision agent...")
                print("📊 Watch the logs above and dual_model_coordination.log for detailed coordination...")

                result = await agent.run(user_input)
                print(f"\n✅ Result:\n{result}")

            except KeyboardInterrupt:
                print("\n\n⏸️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Try rephrasing your request or check the error above")

    except Exception as e:
        print(f"\n❌ Failed to setup dual model agent: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure BOTH models are running:")
        print(f"   - Core LLM: {core_llm_config.api_url}")
        print(f"   - Moondream: {moondream_config.api_url}")
        print("2. Load appropriate models:")
        print(f"   - Core LLM: {core_llm_config.model_name}")
        print(f"   - Moondream: {moondream_config.model_name}")
        print("3. Check MCP server script exists and is executable")
        print("4. Install required dependencies:")
        print("   pip install playwright langchain langchain-openai aiohttp")
        print("   playwright install chromium")
        print("\n💡 TIP: You can run both models in the same LM Studio instance")
        print("or use separate instances on different ports")
        print("\n🆘 If issues persist, check the logs above for specific errors")
        print("📊 Detailed logs are in dual_model_coordination.log")

    finally:
        print("\n🧹 Cleaning up...")
        await agent.cleanup()
        print("👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())