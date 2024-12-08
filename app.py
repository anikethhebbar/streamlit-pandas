import os
import matplotlib
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import io



class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    
    def parse(self, result):
        with st.chat_message("assistant"):
            st.write("Here's what I found:")
            
            # Universal preprocessing to standardize result format
            def standardize_result(raw_result):
                # If result is already in correct format, return as is
                if isinstance(raw_result, dict) and 'type' in raw_result and 'value' in raw_result:
                    return raw_result
                
                # Handle DataFrame input
                if isinstance(raw_result, pd.DataFrame):
                    # Handle missing data and type conversion
                    df = raw_result.copy()
                    for col in df.columns:
                        # Try to convert to numeric, but keep as original if not possible
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            # Fallback to string conversion with safe handling
                            df[col] = df[col].fillna('').astype(str)
                    
                    return {
                        'type': 'dataframe',
                        'value': df
                    }
                
                # Handle Series input
                if isinstance(raw_result, pd.Series):
                    try:
                        # Convert Series to numeric if possible
                        numeric_series = pd.to_numeric(raw_result, errors='ignore')
                        return {
                            'type': 'series',
                            'value': numeric_series
                        }
                    except:
                        # Fallback to string series
                        return {
                            'type': 'series',
                            'value': raw_result.fillna('').astype(str)
                        }
                
                # Handle list input
                if isinstance(raw_result, list):
                    try:
                        # Convert list to DataFrame with smart type inference
                        if any(isinstance(x, (list, tuple)) for x in raw_result):
                            # Pad lists to equal length
                            max_len = max(len(x) if isinstance(x, (list, tuple)) else 1 for x in raw_result)
                            padded_result = []
                            for x in raw_result:
                                if isinstance(x, (list, tuple)):
                                    padded_result.append(list(x) + [None] * (max_len - len(x)))
                                else:
                                    padded_result.append([x] + [None] * (max_len - 1))
                            
                            df = pd.DataFrame(padded_result)
                            
                            # Attempt to convert columns to numeric
                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            
                            return {
                                'type': 'dataframe',
                                'value': df
                            }
                        else:
                            # Simple list conversion
                            return {
                                'type': 'text',
                                'value': ', '.join(map(str, raw_result))
                            }
                    except Exception as e:
                        # Fallback to text representation
                        return {
                            'type': 'text',
                            'value': str(raw_result)
                        }
                
                # Handle dictionary input
                if isinstance(raw_result, dict):
                    # Ensure type and value keys exist
                    if 'type' not in raw_result:
                        # Try to infer type based on value
                        if isinstance(raw_result.get('value'), pd.DataFrame):
                            raw_result['type'] = 'dataframe'
                        elif isinstance(raw_result.get('value'), (dict, list)):
                            # Convert to string representation for complex nested structures
                            raw_result['type'] = 'text'
                            raw_result['value'] = str(raw_result.get('value', ''))
                        else:
                            raw_result['type'] = 'text'
                    
                    # Ensure value exists
                    if 'value' not in raw_result:
                        raw_result['value'] = str(raw_result)
                    
                    return raw_result
                
                # Handle other types by converting to text
                return {
                    'type': 'text',
                    'value': str(raw_result)
                }

            # Standardize the result format
            result = standardize_result(result)

            # Handle the result based on type
            try:
                if result['type'] == "dataframe":
                    st.write("I've organized the data into a nice table for you:")
                    if isinstance(result['value'], pd.DataFrame):
                        result['value'] = result['value'].astype(str)
                    st.dataframe(result['value'])
                elif result['type'] == 'plot':
                    st.write("I've created a visualization to help explain the data:")
                    
                    try:
                        # Check if the plot is a base64 encoded string
                        if isinstance(result['value'], str) and result['value'].startswith('data:image/png;base64,'):
                            import base64
                            from PIL import Image
                            
                            # Remove the data URL prefix
                            base64_str = result['value'].split(',')[1]
                            
                            # Decode the base64 string
                            image_bytes = base64.b64decode(base64_str)
                            
                            # Open the image
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Display the image in Streamlit
                            st.image(image)
                        
                        # If it's a matplotlib figure
                        elif isinstance(result['value'], matplotlib.figure.Figure):
                            st.pyplot(result['value'])
                            matplotlib.pyplot.close(result['value'])
                        
                        # For other plot types
                        else:
                            st.write(f"Unexpected plot type: {type(result['value'])}")
                            st.write(result['value'])
                    
                    except Exception as e:
                        st.error(f"Error displaying plot: {str(e)}")
                        st.write("Full plot result:", result)
                elif result['type'] == 'series':
                    st.write("Here are the values:")
                    # Create DataFrame with index and handle scalar values
                    if isinstance(result['value'], pd.Series):
                        display_df = pd.DataFrame(result['value'])
                    else:
                        display_df = pd.DataFrame({'Value': result['value']}, index=[0])
                    st.dataframe(display_df)
                elif result['type'] == 'number':
                    if isinstance(result['value'], dict):
                        formatted_values = []
                        for key, value in result['value'].items():
                            formatted_values.append(f"{key}: {value}")
                        st.write("\n".join(formatted_values))
                    elif isinstance(result['value'], (list, tuple)):
                        st.write(", ".join(map(str, result['value'])))
                    else:
                        st.write(result['value'])
                else:
                    st.write(result['value'])
            except Exception as e:
                st.error(f"Error displaying result: {str(e)}")
                st.write("Raw result:", result)
                
            st.write("Is there anything specific about this data you'd like me to explain further?")
        return


def setup():
    st.header("Chat with your datasets!", anchor=False, divider="red")

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def main():
    setup()
    
    # Add API key input in sidebar
    with st.sidebar:
        groq_api_key = st.text_input("Enter your Groq API key", type="password")
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue")
            st.stop()
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    dataset = st.file_uploader("Upload your csv or xlsx file", type=['csv','xlsx'])
    if not dataset: st.stop()
    
    try:
        if dataset.name.endswith('.csv'):
            df = pd.read_csv(dataset, low_memory=False, encoding_errors='replace')
        else:  # xlsx
            df = pd.read_excel(dataset)
            
        # Try to convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                continue
                
    except UnicodeDecodeError:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(dataset, low_memory=False, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            st.error("Unable to read the file. Please check the file encoding.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()
    
    with st.chat_message("assistant"):        
        st.write("Let me show you a preview of your data:")
        try:
            preview_df = df.head().astype(str)
            st.dataframe(preview_df)
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")
            st.write("Raw preview data:", df.head())
    
    try:
        llm = ChatGroq(model_name="llama-3.2-3b-preview", api_key=os.environ["GROQ_API_KEY"])
        sdf = SmartDataframe(
            df, 
            config={
                "llm": llm, 
                "conversational": True,
                "response_parser": OutputParser,
                "enable_safe_mode": False,
                "custom_whitelisted_dependencies": ["matplotlib.pyplot"],
                "verbose": True  # Increased verbosity for debugging
            }
        )
    except Exception as e:
        st.error(f"Error initializing AI components: {str(e)}")
        st.stop()
    
    # Accept user input using chat interface
    if prompt := st.chat_input("What would you like to know about your data?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = sdf.chat(prompt)
            
            # Show the code in an expandable section
            with st.chat_message("assistant"):
                if hasattr(sdf, 'last_code_executed') and sdf.last_code_executed:
                    st.divider()
                    with st.expander("Click to see the code executed behind the scenes"):
                        st.code(sdf.last_code_executed)
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
            if hasattr(sdf, 'last_code_executed') and sdf.last_code_executed:
                with st.expander("Last executed code before error"):
                    st.code(sdf.last_code_executed)


if __name__ == '__main__':
    load_dotenv(override=True)
    matplotlib.use("Agg", force=True)
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")