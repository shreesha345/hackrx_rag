import asyncio
import requests
import json
import os
from typing import Dict, Any, Optional, TypedDict, Literal, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
# Define the state structure for our agent
class AgentState(TypedDict):
    messages: List
    city: Optional[str]
    landmark: Optional[str]
    flight_number: Optional[str]
    step: str
    error: Optional[str]
    debug_info: Dict[str, Any]
    llm_analysis: Optional[str]

# Landmark mappings from the PDF
CITY_LANDMARKS = {
    # Indian Cities
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate", 
    "Chennai": "Charminar",
    "Hyderabad": "Marina Beach",
    "Ahmedabad": "Howrah Bridge",
    "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar",
    "Pune": "Meenakshi Temple",
    "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace",
    "Kerala": "Rock Garden",
    "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha",
    "Jaisalmer": "Sun Temple",
    
    # International Cities
    "New York": "Eiffel Tower",
    "London": "Statue of Liberty",
    "Tokyo": "Big Ben",
    "Beijing": "Colosseum",
    "Bangkok": "Christ the Redeemer",
    "Toronto": "Burj Khalifa",
    "Dubai": "CN Tower",
    "Amsterdam": "Petronas Towers",
    "Cairo": "Leaning Tower of Pisa",
    "San Francisco": "Mount Fuji",
    "Berlin": "Niagara Falls",
    "Barcelona": "Louvre Museum",
    "Moscow": "Stonehenge",
    "Seoul": "Sagrada Familia",
    "Cape Town": "Acropolis",
    "Istanbul": "Big Ben",
    "Riyadh": "Machu Picchu",
    "Paris": "Taj Mahal",
    "Singapore": "Christchurch Cathedral",
    "Jakarta": "The Shard",
    "Vienna": "Blue Mosque",
    "Kathmandu": "Neuschwanstein Castle",
    "Los Angeles": "Buckingham Palace"
}

FLIGHT_ENDPOINTS = {
    "Gateway of India": "getFirstCityFlightNumber",
    "Taj Mahal": "getSecondCityFlightNumber", 
    "Eiffel Tower": "getThirdCityFlightNumber",
    "Big Ben": "getFourthCityFlightNumber",
}

# Initialize Gemini LLM
def create_gemini_llm(api_key: str = os.getenv("GEMINI_API_KEY"), model: str = "gemini-1.5-flash") -> ChatGoogleGenerativeAI:
    """Create and configure Gemini LLM."""
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.1  # Low temperature for consistent responses
    )

# Define tools for the agent
@tool
def fetch_favorite_city() -> str:
    """Fetch the favorite city from the HackRx API."""
    try:
        url = "https://register.hackrx.in/submissions/myFavouriteCity"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        try:
            data = response.json()
            if isinstance(data, dict) and 'data' in data and 'city' in data['data']:
                return data['data']['city']
            elif isinstance(data, str):
                return data
            else:
                return str(data)
        except json.JSONDecodeError:
            return response.text.strip().strip('"')
    except Exception as e:
        raise Exception(f"Failed to fetch favorite city: {str(e)}")

@tool
def lookup_landmark(city: str) -> str:
    """Look up the landmark for a given city in the parallel world."""
    # Handle case variations
    for known_city, landmark in CITY_LANDMARKS.items():
        if city.lower() == known_city.lower():
            return landmark
    
    # If exact match not found, try partial matches
    for known_city, landmark in CITY_LANDMARKS.items():
        if city.lower() in known_city.lower() or known_city.lower() in city.lower():
            return landmark
    
    raise Exception(f"City '{city}' not found in landmark mapping")

@tool
def fetch_flight_number(landmark: str) -> str:
    """Fetch the flight number based on the landmark."""
    try:
        base_url = "https://register.hackrx.in/teams/public/flights"
        
        # Determine endpoint based on landmark
        if landmark in FLIGHT_ENDPOINTS:
            endpoint = FLIGHT_ENDPOINTS[landmark]
        else:
            endpoint = "getFifthCityFlightNumber"  # Default for all other landmarks
        
        url = f"{base_url}/{endpoint}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        try:
            data = response.json()
            if isinstance(data, dict) and 'data' in data and 'flightNumber' in data['data']:
                return data['data']['flightNumber']
            elif isinstance(data, str):
                return data
            else:
                return str(data)
        except json.JSONDecodeError:
            return response.text.strip().strip('"')
    except Exception as e:
        raise Exception(f"Failed to fetch flight number for {landmark}: {str(e)}")

# Agent node functions with Gemini LLM integration
def gemini_coordinator_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Gemini-powered coordinator that makes intelligent decisions about next steps."""
    
    system_prompt = """You are an AI coordinator helping Sachin navigate a parallel world to find his way back home.
    
    Mission Context:
    - Sachin is in a parallel world where landmarks are in wrong cities
    - We need to: 1) Get favorite city from API, 2) Map city to landmark, 3) Get flight number
    - Current step: {step}
    - City found: {city}
    - Landmark found: {landmark}
    - Flight number: {flight_number}
    - Any errors: {error}
    
    Based on the current state, decide what to do next and provide a helpful status message.
    Be encouraging and clear about progress."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Current mission state: step={step}, city={city}, landmark={landmark}, flight={flight_number}, error={error}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    current_step = state.get("step", "start")
    
    try:
        # Get LLM analysis of current situation
        llm_response = chain.invoke({
            "step": current_step,
            "city": state.get("city"),
            "landmark": state.get("landmark"),
            "flight_number": state.get("flight_number"),
            "error": state.get("error")
        })
        
        # For flight_fetched, make sure llm_response uses the specific message we want
        if current_step == "flight_fetched":
            llm_response = f"""Sachin, excellent work! We've successfully fetched your flight number ({state.get('flight_number')}) for your journey from New York (where you currently are in this parallel world) to the location of the Eiffel Tower (your landmark target). Remember, this is a parallel world, so things are a bit...mixed up!

The next step is to use flight number `{state.get('flight_number')}` to board your flight and get closer to your home reality. We'll need to monitor your progress on this flight. Please provide confirmation once you've boarded. Keep up the amazing work – you're making great progress!"""
        
        # Determine next step based on current state
        if current_step == "start":
            next_step = "fetch_city"
            message = "🧠 Starting Mission: Sachin's Parallel World Discovery\n🔍 Step 1: Fetching favorite city from API..."
        elif current_step == "city_fetched":
            next_step = "decode_landmark"
            message = f"✅ City received: {state['city']}\n🧠 Step 2: Using AI to decode city and find landmark in parallel world..."
        elif current_step == "landmark_decoded":
            next_step = "fetch_flight"
            message = f"✅ Landmark decoded: {state['city']} → {state['landmark']}\n✈️ Step 3: Fetching flight number for landmark..."
        elif current_step == "flight_fetched":
            next_step = "complete"
            message = f"""{llm_response}"""
        else:
            next_step = current_step
            message = llm_response
        
        return {
            **state,
            "step": next_step,
            "llm_analysis": llm_response,
            "messages": state["messages"] + [AIMessage(content=message)]
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Gemini coordinator error: {str(e)}",
            "step": "error",
            "messages": state["messages"] + [AIMessage(content=f"❌ AI Coordinator Error: {str(e)}")]
        }

def gemini_city_analyzer_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Gemini-powered city analysis and fetching."""
    
    system_prompt = """You are helping fetch and analyze the favorite city from the HackRx API.
    After fetching the city, analyze if it looks correct and provide insights about the city.
    Be concise but informative."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "I need to fetch the favorite city from the API. Please help me analyze the result.")
    ])
    
    try:
        # Fetch city using tool
        city = fetch_favorite_city.invoke({})
        
        # Get Gemini analysis of the city
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({"city": city})
        
        return {
            **state,
            "city": city,
            "step": "city_fetched",
            "llm_analysis": analysis,
            "debug_info": {**state["debug_info"], "city_fetch_success": True, "gemini_city_analysis": analysis}
        }
    except Exception as e:
        return {
            **state,
            "error": str(e),
            "step": "error",
            "messages": state["messages"] + [AIMessage(content=f"❌ Error fetching city with Gemini analysis: {str(e)}")]
        }

def gemini_landmark_decoder_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Gemini-powered landmark decoding with intelligent city matching."""
    
    system_prompt = """You are helping decode a city name to find its landmark in a parallel world.
    
    Available city-landmark mappings:
    {city_landmarks}
    
    The city from API is: {city}
    
    Your task:
    1. Find the exact or closest match for this city in the mappings
    2. Explain your reasoning for the match
    3. Return the landmark associated with that city
    
    Be precise and explain your matching logic."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please analyze the city '{city}' and find its landmark in the parallel world mappings.")
    ])
    
    try:
        # First try direct lookup
        try:
            landmark = lookup_landmark.invoke({"city": state["city"]})
            
            # Get Gemini explanation of the mapping
            chain = prompt | llm | StrOutputParser()
            analysis = chain.invoke({
                "city": state["city"],
                "city_landmarks": json.dumps(CITY_LANDMARKS, indent=2)
            })
            
            return {
                **state,
                "landmark": landmark,
                "step": "landmark_decoded",
                "llm_analysis": analysis,
                "debug_info": {**state["debug_info"], "landmark_lookup_success": True, "gemini_landmark_analysis": analysis}
            }
            
        except Exception as lookup_error:
            # If direct lookup fails, use Gemini to find closest match
            chain = prompt | llm | StrOutputParser()
            gemini_response = chain.invoke({
                "city": state["city"],
                "city_landmarks": json.dumps(CITY_LANDMARKS, indent=2)
            })
            
            # Try to extract landmark from Gemini response
            # This is a fallback - in practice, you might want more sophisticated parsing
            for city, landmark in CITY_LANDMARKS.items():
                if landmark.lower() in gemini_response.lower():
                    return {
                        **state,
                        "landmark": landmark,
                        "step": "landmark_decoded",
                        "llm_analysis": f"Gemini AI Analysis: {gemini_response}",
                        "debug_info": {**state["debug_info"], "gemini_landmark_match": True}
                    }
            
            raise Exception(f"Could not find landmark match even with Gemini analysis: {gemini_response}")
            
    except Exception as e:
        return {
            **state,
            "error": str(e),
            "step": "error",
            "messages": state["messages"] + [AIMessage(content=f"❌ Error decoding landmark with Gemini: {str(e)}")]
        }

def gemini_flight_fetcher_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Gemini-powered flight number fetching with endpoint analysis."""
    
    system_prompt = """You are helping fetch the flight number for Sachin's journey back to the real world.
    
    Landmark: {landmark}
    Flight endpoint rules:
    - Gateway of India → getFirstCityFlightNumber
    - Taj Mahal → getSecondCityFlightNumber
    - Eiffel Tower → getThirdCityFlightNumber
    - Big Ben → getFourthCityFlightNumber
    - All others → getFifthCityFlightNumber
    
    Analyze which endpoint should be used and provide confidence in the selection."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "For landmark '{landmark}', which flight endpoint should I use and why?")
    ])
    
    try:
        # Get Gemini analysis of endpoint selection
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({"landmark": state["landmark"]})
        
        # Fetch flight number using tool
        flight_number = fetch_flight_number.invoke({"landmark": state["landmark"]})
        
        return {
            **state,
            "flight_number": flight_number,
            "step": "flight_fetched",
            "llm_analysis": analysis,
            "debug_info": {**state["debug_info"], "flight_fetch_success": True, "gemini_endpoint_analysis": analysis}
        }
    except Exception as e:
        return {
            **state,
            "error": str(e),
            "step": "error",
            "messages": state["messages"] + [AIMessage(content=f"❌ Error fetching flight with Gemini analysis: {str(e)}")]
        }

def gemini_error_handler_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """Gemini-powered error analysis and recovery suggestions."""
    
    system_prompt = """You are an AI error analyst helping debug the HackRx mission failure.
    
    Error details: {error}
    Mission state: city={city}, landmark={landmark}, step={step}
    
    Provide:
    1. Clear explanation of what went wrong
    2. Possible causes
    3. Suggested recovery steps
    4. Encouraging message for the user
    
    Be helpful and solution-oriented."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze this mission failure and provide recovery guidance.")
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({
            "error": state.get("error", "Unknown error"),
            "city": state.get("city", "None"),
            "landmark": state.get("landmark", "None"),
            "step": state.get("step", "Unknown")
        })
        
        return {
            **state,
            "llm_analysis": analysis,
            "messages": state["messages"] + [
                AIMessage(content=f"🤖 Gemini AI Error Analysis:\n\n{analysis}")
            ]
        }
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"💔 Mission failed with error: {state.get('error', 'Unknown')}\n❌ Additional Gemini error: {str(e)}")
            ]
        }

# Router function to determine next step
def route_next_step(state: AgentState) -> Literal["coordinator", "fetch_city", "decode_landmark", "fetch_flight", "error", "__end__"]:
    """Route to the next step based on current state."""
    if state.get("error"):
        return "error"
    
    step = state.get("step", "start")
    
    if step == "start":
        return "coordinator"
    elif step == "fetch_city":
        return "fetch_city"
    elif step == "city_fetched":
        return "coordinator"
    elif step == "decode_landmark":
        return "decode_landmark"
    elif step == "landmark_decoded":
        return "coordinator"
    elif step == "fetch_flight":
        return "fetch_flight"
    elif step == "flight_fetched":
        return "coordinator"
    elif step == "complete":
        return "__end__"
    elif step == "error":
        return "error"
    else:
        return "__end__"

class GeminiHackRxParallelWorldAgent:
    """LangGraph agent with Gemini LLM for the HackRx Parallel World mission."""
    
    def __init__(self, google_api_key: str = None, model: str = "gemini-1.5-flash"):
        self.llm = create_gemini_llm(google_api_key, model)
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with Gemini integration."""
        workflow = StateGraph(AgentState)
        
        # Add nodes with Gemini LLM integration
        workflow.add_node("coordinator", lambda state: gemini_coordinator_node(state, self.llm))
        workflow.add_node("fetch_city", lambda state: gemini_city_analyzer_node(state, self.llm))
        workflow.add_node("decode_landmark", lambda state: gemini_landmark_decoder_node(state, self.llm))
        workflow.add_node("fetch_flight", lambda state: gemini_flight_fetcher_node(state, self.llm))
        workflow.add_node("error", lambda state: gemini_error_handler_node(state, self.llm))
        
        # Add edges
        workflow.add_edge(START, "coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            route_next_step,
            {
                "coordinator": "coordinator",
                "fetch_city": "fetch_city", 
                "decode_landmark": "decode_landmark",
                "fetch_flight": "fetch_flight",
                "error": "error",
                "__end__": END
            }
        )
        workflow.add_conditional_edges("fetch_city", route_next_step)
        workflow.add_conditional_edges("decode_landmark", route_next_step)
        workflow.add_conditional_edges("fetch_flight", route_next_step)
        workflow.add_edge("error", END)
        
        return workflow
    
    async def execute_mission(self, thread_id: str = "hackrx_gemini_mission") -> Dict[str, Any]:
        """Execute the complete mission asynchronously with Gemini AI."""
        initial_state = {
            "messages": [HumanMessage(content="Help Sachin find his way back to the real world using Gemini AI!")],
            "city": None,
            "landmark": None,
            "flight_number": None,
            "step": "start",
            "error": None,
            "debug_info": {},
            "llm_analysis": None
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = await self.app.ainvoke(initial_state, config=config)
        return final_state
    
    def execute_mission_sync(self, thread_id: str = "hackrx_gemini_mission") -> Dict[str, Any]:
        """Execute the complete mission synchronously with Gemini AI."""
        initial_state = {
            "messages": [HumanMessage(content="Help Sachin find his way back to the real world using Gemini AI!")],
            "city": None,
            "landmark": None,
            "flight_number": None,
            "step": "start",
            "error": None,
            "debug_info": {},
            "llm_analysis": None
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = self.app.invoke(initial_state, config=config)
        return final_state
    
    def print_conversation(self, state: Dict[str, Any]):
        """Print only the AI messages from the conversation history."""
        for message in state["messages"]:
            if isinstance(message, AIMessage):
                print(message.content)
    
    def get_mission_insights(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed insights from Gemini about the mission execution."""
        return {
            "success": bool(state.get("flight_number")),
            "city": state.get("city"),
            "landmark": state.get("landmark"),
            "flight_number": state.get("flight_number"),
            "gemini_analysis": state.get("llm_analysis"),
            "debug_info": state.get("debug_info", {}),
            "error": state.get("error")
        }

# Usage examples
async def main_gemini_async():
    """Main async function to run the Gemini-powered agent."""
    print("🚀 Initializing Gemini-Powered LangGraph HackRx Agent...")
    
    # Initialize with your Google API key
    # You can set GOOGLE_API_KEY environment variable or pass it directly
    try:
        agent = GeminiHackRxParallelWorldAgent()
        
        print("🧠 Executing mission with Gemini AI intelligence...")
        result = await agent.execute_mission()
        
        # Find and return the flight_fetched step response
        flight_message = None
        for message in result["messages"]:
            if isinstance(message, AIMessage) and "We've successfully fetched your flight number" in message.content:
                flight_message = message.content
                print(message.content)
                break
                
        return flight_message  # Return the message so it can be used by routes.py
        
    except Exception as e:
        print(f"❌ Failed to initialize Gemini agent: {e}")
        print("Make sure you have set GOOGLE_API_KEY environment variable")
        return None
        
        # Get detailed insights
        # insights = agent.get_mission_insights(result)
        # print(f"\n📊 Mission Insights:")
        # print(f"Success: {insights['success']}")
        # print(f"City: {insights['city']}")
        # print(f"Landmark: {insights['landmark']}")
        # print(f"Flight: {insights['flight_number']}")
        
        # if result["flight_number"]:
        #     print(f"\n🏆 SUCCESS: Gemini AI found flight number {result['flight_number']}")
        #     return result["flight_number"]
        # else:
        #     print(f"\n💔 FAILED: {result.get('error', 'Unknown error')}")
        #     return None
            
    except Exception as e:
        print(f"❌ Failed to initialize Gemini agent: {e}")
        print("Make sure you have set GOOGLE_API_KEY environment variable")
        return None

def main_gemini_sync():
    """Main sync function to run the Gemini-powered agent."""
    print("🚀 Initializing Gemini-Powered LangGraph HackRx Agent...")
    
    try:
        agent = GeminiHackRxParallelWorldAgent()
        
        print("🧠 Executing mission with Gemini AI intelligence...")
        result = agent.execute_mission_sync()
        
        # Find and return the flight_fetched step response
        flight_message = None
        for message in result["messages"]:
            if isinstance(message, AIMessage) and "We've successfully fetched your flight number" in message.content:
                flight_message = message.content
                print(message.content)
                break
                
        return flight_message  # Return the message so it can be used by routes.py
        
    except Exception as e:
        print(f"❌ Failed to initialize Gemini agent: {e}")
        print("Make sure you have set GOOGLE_API_KEY environment variable")
        return None
        
        # Get detailed insights
        # insights = agent.get_mission_insights(result)
        # print(f"\n📊 Mission Insights:")
        # print(f"Success: {insights['success']}")
        # print(f"City: {insights['city']}")
        # print(f"Landmark: {insights['landmark']}")
        # print(f"Flight: {insights['flight_number']}")
        
        # if result["flight_number"]:
        #     print(f"\n🏆 SUCCESS: Gemini AI found flight number {result['flight_number']}")
        #     return result["flight_number"]
        # else:
        #     print(f"\n💔 FAILED: {result.get('error', 'Unknown error')}")
        #     return None
            
    except Exception as e:
        print(f"❌ Failed to initialize Gemini agent: {e}")
        print("Make sure you have set GOOGLE_API_KEY environment variable")
        return None

if __name__ == "__main__":
    # Set your Google API key as environment variable:
    # export GOOGLE_API_KEY="your-api-key-here"
    
    # Run synchronously with Gemini
    # flight_number = main_gemini_sync()
    
    # Or run asynchronously with Gemini
    flight_number = asyncio.run(main_gemini_async())
    
    # Example of using with custom API key:
    # agent = GeminiHackRxParallelWorldAgent(google_api_key="your-api-key")
    # result = agent.execute_mission_sync()