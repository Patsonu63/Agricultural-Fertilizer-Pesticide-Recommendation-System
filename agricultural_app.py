"""
SUMIT PVT LTD - Agricultural Fertilizer & Pesticide Recommendation System
Built with LangChain and LangGraph

This application helps farmers determine optimal fertilizer and pesticide
usage based on crop type, soil conditions, and pest detection.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import RunnableLambda
import langchain_core.messages as messages
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

# Configure API keys (ensure you have set these in your .env file)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key")

# Define a database of crops, fertilizers, and pesticides
CROP_DATABASE = {
    "rice": {
        "fertilizers": {
            "urea": "175-200 kg/ha in multiple doses",
            "dap": "100 kg/ha as basal application",
            "potash": "60 kg/ha split in two applications"
        },
        "common_pests": ["stem borer", "leaf folder", "brown planthopper", "rice hispa"],
        "soil_requirements": "Clay or clay loam with pH 5.5-6.5",
        "seasons": "Kharif, Rabi (depending on region)"
    },
    "wheat": {
        "fertilizers": {
            "urea": "240-260 kg/ha in 2-3 split doses",
            "dap": "150 kg/ha as basal dose",
            "potash": "60 kg/ha as basal application"
        },
        "common_pests": ["aphid", "termite", "pink stem borer", "army worm"],
        "soil_requirements": "Loam or clay loam with pH 6.0-7.5",
        "seasons": "Rabi"
    },
    "cotton": {
        "fertilizers": {
            "urea": "175-200 kg/ha in split applications",
            "dap": "100 kg/ha as basal application",
            "potash": "50 kg/ha in split doses"
        },
        "common_pests": ["bollworm", "whitefly", "jassid", "thrips", "aphid"],
        "soil_requirements": "Well-drained black cotton soil, loamy soil with pH 6.0-8.0",
        "seasons": "Kharif"
    },
    "sugarcane": {
        "fertilizers": {
            "urea": "300-325 kg/ha in multiple doses",
            "dap": "125 kg/ha as basal dose",
            "potash": "120 kg/ha in split applications"
        },
        "common_pests": ["early shoot borer", "top borer", "pyrilla", "white grub"],
        "soil_requirements": "Deep rich loamy soil with pH 6.5-7.5",
        "seasons": "Year-round (varies by region)"
    },
    "maize": {
        "fertilizers": {
            "urea": "250-300 kg/ha in split applications",
            "dap": "150 kg/ha as basal dose",
            "potash": "80 kg/ha as basal application"
        },
        "common_pests": ["stem borer", "army worm", "shoot fly", "leaf hopper"],
        "soil_requirements": "Well-drained loamy soil with pH 6.5-7.5",
        "seasons": "Kharif, Rabi, Spring"
    }
}

PESTICIDE_DATABASE = {
    "stem borer": ["Cartap hydrochloride 4G", "Chlorantraniliprole 0.4% GR", "Fipronil 0.3% GR"],
    "leaf folder": ["Chlorantraniliprole 18.5% SC", "Flubendiamide 20% WG", "Thiamethoxam 25% WG"],
    "brown planthopper": ["Buprofezin 25% SC", "Dinotefuran 20% SG", "Pymetrozine 50% WG"],
    "rice hispa": ["Quinalphos 25% EC", "Chlorpyriphos 20% EC", "Phenthoate 50% EC"],
    "aphid": ["Imidacloprid 17.8% SL", "Thiamethoxam 25% WG", "Acetamiprid 20% SP"],
    "termite": ["Chlorpyriphos 20% EC", "Fipronil 5% SC", "Imidacloprid 30.5% SC"],
    "pink stem borer": ["Chlorantraniliprole 18.5% SC", "Flubendiamide 39.35% SC", "Spinosad 45% SC"],
    "army worm": ["Emamectin benzoate 5% SG", "Lambda-cyhalothrin 5% EC", "Indoxacarb 14.5% SC"],
    "bollworm": ["Emamectin benzoate 5% SG", "Spinosad 45% SC", "Chlorantraniliprole 18.5% SC"],
    "whitefly": ["Diafenthiuron 50% WP", "Pyriproxyfen 10% EC", "Spiromesifen 22.9% SC"],
    "jassid": ["Imidacloprid 17.8% SL", "Thiamethoxam 25% WG", "Dinotefuran 20% SG"],
    "thrips": ["Fipronil 5% SC", "Spinosad 45% SC", "Spinetoram 11.7% SC"],
    "early shoot borer": ["Chlorantraniliprole 0.4% GR", "Fipronil 0.3% GR", "Cartap hydrochloride 4G"],
    "top borer": ["Chlorantraniliprole 18.5% SC", "Flubendiamide 39.35% SC", "Lambda-cyhalothrin 5% EC"],
    "pyrilla": ["Buprofezin 25% SC", "Imidacloprid 17.8% SL", "Thiamethoxam 25% WG"],
    "white grub": ["Chlorpyriphos 20% EC", "Phorate 10% CG", "Fipronil 0.3% GR"],
    "shoot fly": ["Imidacloprid 70% WG", "Carbofuran 3% CG", "Thiamethoxam 25% WG"]
}

FERTILIZER_INFO = {
    "urea": {
        "composition": "46% Nitrogen",
        "application_method": "Side dressing, broadcast, or foliar spray",
        "precautions": "Avoid direct seed contact. Apply in moist soil to prevent volatilization. Split application recommended.",
        "benefits": "Provides essential nitrogen for vegetative growth, leaf development, and protein synthesis."
    },
    "dap": {
        "composition": "18% Nitrogen, 46% Phosphorus",
        "application_method": "Basal application or band placement",
        "precautions": "Store in dry place. Don't mix with alkaline materials. Apply before sowing or at the time of sowing.",
        "benefits": "Promotes root development, flowering, and seed formation. Water-soluble and readily available to plants."
    },
    "potash": {
        "composition": "60% Potassium",
        "application_method": "Broadcast application or band placement",
        "precautions": "Apply as per soil test recommendation. Avoid contact with plant foliage in concentrated form.",
        "benefits": "Enhances disease resistance, drought tolerance, and overall plant vigor. Improves quality of produce."
    },
    "npk": {
        "composition": "Varies (common ratios: 10-26-26, 12-32-16, 20-20-0)",
        "application_method": "Basal application or top dressing",
        "precautions": "Apply as per crop requirement. Store in dry place. Keep away from children and foodstuff.",
        "benefits": "Balanced nutrition with multiple nutrients in a single application."
    },
    "ssp": {
        "composition": "16% Phosphorus, 12% Sulfur, 21% Calcium",
        "application_method": "Basal application before sowing",
        "precautions": "Store in dry place. Not compatible with alkaline fertilizers.",
        "benefits": "Slow-release phosphorus source. Also provides sulfur and calcium. Good for acidic soils."
    }
}

# Define output schemas
class SoilAnalysis(BaseModel):
    soil_type: str = Field(description="Type of soil identified")
    ph_level: float = Field(description="pH level of the soil")
    nutrient_levels: Dict[str, str] = Field(description="Nutrient levels in the soil (nitrogen, phosphorus, potassium)")
    recommendations: List[str] = Field(description="Recommendations for soil improvement")

class PestIdentification(BaseModel):
    pest_name: str = Field(description="Name of the identified pest")
    severity: str = Field(description="Severity of infestation (low, medium, high)")
    recommended_pesticides: List[str] = Field(description="List of recommended pesticides")
    application_method: str = Field(description="How to apply the pesticides")
    precautions: List[str] = Field(description="Safety precautions while handling pesticides")

class FertilizerRecommendation(BaseModel):
    crop_name: str = Field(description="Name of the crop")
    growth_stage: str = Field(description="Current growth stage of the crop")
    recommended_fertilizers: Dict[str, str] = Field(description="Recommended fertilizers with dosage")
    application_schedule: List[str] = Field(description="When to apply these fertilizers")
    expected_benefits: List[str] = Field(description="Expected benefits of these fertilizers")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create knowledge base from documents
def create_knowledge_base():
    # In a real application, you would load actual documents here
    # For this example, we'll create sample documents from our database
    documents = []
    
    # Create crop documents
    for crop, data in CROP_DATABASE.items():
        content = f"Crop: {crop}\n"
        content += f"Soil Requirements: {data['soil_requirements']}\n"
        content += f"Growing Seasons: {data['seasons']}\n"
        content += "Fertilizer Requirements:\n"
        for fert, dose in data['fertilizers'].items():
            content += f"- {fert.upper()}: {dose}\n"
        content += "Common Pests:\n"
        for pest in data['common_pests']:
            content += f"- {pest}\n"
            
        documents.append(content)
    
    # Create pesticide documents
    for pest, pesticides in PESTICIDE_DATABASE.items():
        content = f"Pest: {pest}\n"
        content += "Recommended Pesticides:\n"
        for pesticide in pesticides:
            content += f"- {pesticide}\n"
        documents.append(content)
    
    # Create fertilizer documents
    for fert, info in FERTILIZER_INFO.items():
        content = f"Fertilizer: {fert.upper()}\n"
        content += f"Composition: {info['composition']}\n"
        content += f"Application Method: {info['application_method']}\n"
        content += f"Precautions: {info['precautions']}\n"
        content += f"Benefits: {info['benefits']}\n"
        documents.append(content)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

# Create retrieval function
def retrieve_information(query: str, vector_store: VectorStore) -> List[str]:
    docs = vector_store.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]

# Create nodes for LangGraph
def soil_analysis_node(state):
    soil_description = state["soil_description"]
    
    template = """
    You are an agricultural soil expert. Analyze the given soil description and provide detailed information.
    Use the following format for your response:
    
    Soil Description:
    {soil_description}
    
    Based on this information, provide a comprehensive soil analysis.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser(pydantic_object=SoilAnalysis)
    
    chain = (
        prompt 
        | llm
        | parser
    )
    
    result = chain.invoke({"soil_description": soil_description})
    return {"soil_analysis": result}

def pest_identification_node(state):
    pest_description = state["pest_description"]
    crop_type = state.get("crop_type", "unknown")
    
    # Retrieve relevant information from knowledge base
    relevant_info = retrieve_information(f"pest {pest_description} in {crop_type}", state["vector_store"])
    
    template = """
    You are an agricultural pest expert. Identify the pest based on the description and provide control measures.
    
    Pest Description: {pest_description}
    Crop Type: {crop_type}
    
    Additional Information:
    {relevant_info}
    
    Based on this information, provide pest identification and control recommendations.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser(pydantic_object=PestIdentification)
    
    chain = (
        prompt 
        | llm
        | parser
    )
    
    result = chain.invoke({
        "pest_description": pest_description,
        "crop_type": crop_type,
        "relevant_info": "\n".join(relevant_info)
    })
    
    return {"pest_identification": result}

def fertilizer_recommendation_node(state):
    crop_type = state["crop_type"]
    soil_data = state.get("soil_analysis", None)
    growth_stage = state.get("growth_stage", "unknown")
    
    # Retrieve relevant information from knowledge base
    relevant_info = retrieve_information(f"{crop_type} fertilizer requirements {growth_stage}", state["vector_store"])
    
    template = """
    You are an agricultural fertilizer expert at Sumit Pvt Ltd. Recommend fertilizers based on the crop type, soil data, and growth stage.
    
    Crop Type: {crop_type}
    Growth Stage: {growth_stage}
    Soil Data: {soil_data}
    
    Additional Information:
    {relevant_info}
    
    Based on this information, provide fertilizer recommendations focusing on products like urea and DAP.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser(pydantic_object=FertilizerRecommendation)
    
    chain = (
        prompt 
        | llm
        | parser
    )
    
    soil_data_str = str(soil_data) if soil_data else "No soil data available"
    
    result = chain.invoke({
        "crop_type": crop_type,
        "growth_stage": growth_stage,
        "soil_data": soil_data_str,
        "relevant_info": "\n".join(relevant_info)
    })
    
    return {"fertilizer_recommendation": result}

def final_recommendation_node(state):
    crop_type = state["crop_type"]
    soil_analysis = state.get("soil_analysis", None)
    pest_identification = state.get("pest_identification", None)
    fertilizer_recommendation = state.get("fertilizer_recommendation", None)
    
    template = """
    You are an agricultural consultant at Sumit Pvt Ltd. Provide a comprehensive recommendation based on all the data collected.
    
    Crop Type: {crop_type}
    Soil Analysis: {soil_analysis}
    Pest Identification: {pest_identification}
    Fertilizer Recommendation: {fertilizer_recommendation}
    
    Create a comprehensive action plan for the farmer including:
    1. Soil treatment recommendations
    2. Fertilizer application schedule and dosage
    3. Pest control measures
    4. Best practices for the specific crop
    5. Expected outcomes
    
    Make sure to highlight Sumit Pvt Ltd's products (especially urea and DAP) in your recommendations.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm
    
    result = chain.invoke({
        "crop_type": crop_type,
        "soil_analysis": str(soil_analysis) if soil_analysis else "No soil analysis available",
        "pest_identification": str(pest_identification) if pest_identification else "No pest identification available",
        "fertilizer_recommendation": str(fertilizer_recommendation) if fertilizer_recommendation else "No fertilizer recommendation available"
    })
    
    return {"final_recommendation": result.content}

# Create the LangGraph for agricultural recommendation system
def create_ag_recommendation_graph():
    workflow = StateGraph(name="Agricultural Recommendation System")
    
    # Add nodes
    workflow.add_node("soil_analysis", soil_analysis_node)
    workflow.add_node("pest_identification", pest_identification_node)
    workflow.add_node("fertilizer_recommendation", fertilizer_recommendation_node)
    workflow.add_node("final_recommendation", final_recommendation_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "soil_analysis",
        lambda x: "pest_identification" if x.get("pest_description") else "fertilizer_recommendation",
    )
    workflow.add_edge("pest_identification", "fertilizer_recommendation")
    workflow.add_edge("fertilizer_recommendation", "final_recommendation")
    
    # Add entry and exit points
    workflow.set_entry_point("soil_analysis")
    workflow.add_edge("final_recommendation", END)
    
    # Compile the graph
    return workflow.compile()

# Create Streamlit application
def create_streamlit_app():
    st.set_page_config(
        page_title="Sumit Pvt Ltd - Agricultural Recommendation System",
        page_icon="🌾",
        layout="wide"
    )
    
    # Create knowledge base
    vector_store = create_knowledge_base()
    
    # Create graph
    ag_graph = create_ag_recommendation_graph()
    
    # Header
    st.title("🌾 Sumit Pvt Ltd - Agricultural Recommendation System")
    st.markdown("""
    Welcome to the intelligent agricultural recommendation system powered by LangChain and LangGraph.
    This system helps you optimize your crop management with personalized recommendations for fertilizers and pesticides.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Soil & Crop Info", "Pest Management", "Recommendations"])
    
    with tab1:
        st.header("Soil & Crop Information")
        
        # Input fields
        crop_type = st.selectbox(
            "Select Crop Type",
            options=list(CROP_DATABASE.keys()),
            index=0
        )
        
        growth_stage = st.select_slider(
            "Select Growth Stage",
            options=["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"],
            value="Vegetative"
        )
        
        soil_description = st.text_area(
            "Describe your soil (color, texture, drainage, previous issues, etc.)",
            height=150,
            placeholder="Example: The soil is reddish-brown, slightly clayey with moderate drainage. It gets waterlogged after heavy rain and has shown signs of nutrient deficiency in previous seasons."
        )
    
    with tab2:
        st.header("Pest Management")
        
        pest_description = st.text_area(
            "Describe any pest issues you're experiencing (optional)",
            height=150,
            placeholder="Example: Small green insects on the underside of leaves causing yellowing and curling. Some leaves have small holes and the plant growth appears stunted."
        )
        
        # Display common pests for selected crop
        if crop_type in CROP_DATABASE:
            st.subheader(f"Common Pests for {crop_type.capitalize()}")
            for pest in CROP_DATABASE[crop_type]["common_pests"]:
                st.write(f"• {pest.capitalize()}")
    
    with tab3:
        st.header("Get Recommendations")
        
        if st.button("Generate Recommendations", type="primary"):
            if not soil_description:
                st.warning("Please provide soil description for better recommendations.")
                return
            
            # Show spinner during processing
            with st.spinner("Analyzing data and generating recommendations..."):
                # Prepare initial state
                initial_state = {
                    "crop_type": crop_type,
                    "growth_stage": growth_stage,
                    "soil_description": soil_description,
                    "pest_description": pest_description if pest_description else None,
                    "vector_store": vector_store
                }
                
                # Run the graph
                result = ag_graph.invoke(initial_state)
                
                # Display results
                st.success("Analysis complete!")
                
                # Display final recommendation
                st.subheader("📋 Comprehensive Recommendation")
                st.markdown(result.get("final_recommendation", "No recommendation generated"))
                
                # Create expandable sections for detailed results
                with st.expander("Soil Analysis Details"):
                    if "soil_analysis" in result:
                        soil_data = result["soil_analysis"]
                        st.write(f"**Soil Type:** {soil_data.soil_type}")
                        st.write(f"**pH Level:** {soil_data.ph_level}")
                        
                        st.write("**Nutrient Levels:**")
                        for nutrient, level in soil_data.nutrient_levels.items():
                            st.write(f"- {nutrient.capitalize()}: {level}")
                        
                        st.write("**Recommendations for Soil Improvement:**")
                        for rec in soil_data.recommendations:
                            st.write(f"- {rec}")
                    else:
                        st.write("No soil analysis data available")
                
                if pest_description:
                    with st.expander("Pest Management Details"):
                        if "pest_identification" in result:
                            pest_data = result["pest_identification"]
                            st.write(f"**Identified Pest:** {pest_data.pest_name}")
                            st.write(f"**Infestation Severity:** {pest_data.severity}")
                            
                            st.write("**Recommended Pesticides:**")
                            for pesticide in pest_data.recommended_pesticides:
                                st.write(f"- {pesticide}")
                            
                            st.write(f"**Application Method:** {pest_data.application_method}")
                            
                            st.write("**Safety Precautions:**")
                            for precaution in pest_data.precautions:
                                st.write(f"- {precaution}")
                        else:
                            st.write("No pest identification data available")
                
                with st.expander("Fertilizer Recommendation Details"):
                    if "fertilizer_recommendation" in result:
                        fert_data = result["fertilizer_recommendation"]
                        st.write(f"**Crop:** {fert_data.crop_name}")
                        st.write(f"**Growth Stage:** {fert_data.growth_stage}")
                        
                        st.write("**Recommended Fertilizers:**")
                        for fert, dose in fert_data.recommended_fertilizers.items():
                            st.write(f"- {fert.upper()}: {dose}")
                        
                        st.write("**Application Schedule:**")
                        for schedule in fert_data.application_schedule:
                            st.write(f"- {schedule}")
                        
                        st.write("**Expected Benefits:**")
                        for benefit in fert_data.expected_benefits:
                            st.write(f"- {benefit}")
                    else:
                        st.write("No fertilizer recommendation data available")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Sumit Pvt Ltd. All rights reserved.")
    st.markdown("Powered by LangChain and LangGraph")

# Run the Streamlit application
if __name__ == "__main__":
    create_streamlit_app()
