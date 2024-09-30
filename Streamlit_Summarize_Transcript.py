import streamlit as st
from groq import Groq
import os
import time
import random
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Groq client (move to global scope)
API_KEY = "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"
os.environ["GROQ_API_KEY"] = API_KEY
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


chunk_size = 4000
chunk_overlap = 0
model_selected = "mixtral-8x7b-32768"


# --- Streamlit App ---
st.set_page_config(page_title="Summarize Transcript", page_icon="🌍", layout="wide")
st.title("Summarize Transcript")

# Use session state to persist the transcript and question
if "user_transcript" not in st.session_state:
    st.session_state.user_transcript = ""
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

transcript = st.text_area("Paste your transcript here:", height=250, value=st.session_state.user_transcript)
st.session_state.user_transcript = transcript  # Update session state

question = st.text_input("Ask a specific question about the transcript (optional):", value=st.session_state.user_question)
st.session_state.user_question = question  # Update session state


if st.button("Summarize"):
    if transcript:
        # Set question_to_answer based on user input
        question_to_answer = f"Finally, after displaying your summary, your last task will be to answer this question based on the transcript: {question}" if question else ""

        # Process the transcript and display the summary
        with st.spinner('Summarizing...'):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_text(transcript) 


            def text_summarizer_agent(model_input):

                # Summarizes multiple chunks of text using the Groq model.
                summaries = []

                # if there is more than a specific number (25) of chunks we will only treat 20 chunks in more details on top of the first 5 chunks
                # we need this otherwise the final summary cannot be generated since there are too many tokens in the final prompt
                if len(split_docs) >= 90:
                    selected_numbers = 0
                elif len(split_docs) >= 50 and len(split_docs) < 90:
                    # Always take the first 5 chunks
                    selected_numbers = list(range(5))
                    # Randomly choose 20 from the remaining chunks
                    selected_numbers.extend(random.sample(range(5, len(split_docs)), 5)) 
                elif len(split_docs) > 20 and len(split_docs) < 50:
                    # Always take the first 5 chunks
                    selected_numbers = list(range(5))
                    # Randomly choose 20 from the remaining chunks
                    selected_numbers.extend(random.sample(range(5, len(split_docs)), 20))
                else:
                    selected_numbers = range(0, len(split_docs) + 1)


                for i, chunk in enumerate(split_docs):
                    detailed_summary = i in selected_numbers  # Determine if detailed

                    if detailed_summary:
                        tailored_prompt = f"""
                            Please summarize the following text, aiming to capture the most important points and central ideas. \
                            
                            The summary should be concise and easy to understand, while still providing a clear and accurate overview of the original text. \
                            
                            Do not output anything else than the summary. Do not write an explanation of your thought process/introduction and do not write a conclusion: ONLY the summary.\
                            
                            Do not start with 'Here is the summary:' or any starting/introduction of the sort. Just output the result of your analysis.\

                            Here's the transcript to analyze and summarize: {chunk}
                        """
                    else:
                        tailored_prompt = f"""
                            Analyze the following transcript and extract any statements or ideas that convey profound wisdom, valuable life lessons, or insightful perspectives. \
                            
                            As an output, write the shortest summary you can possibly write to capture the essence of these ideas. Stay factual and only summarize the content in this transcript. \
                            
                            Do not output anything else than the summary. Do not write an explanation of your thought process/introduction and do not write a conclusion: ONLY the summary.\
                            
                            Do not start with 'Here is the summary:' or any starting/introduction of the sort. Just output the result of your analysis.\

                            Here's the transcript to analyze and summarize: {chunk}
                        """
                    
                    # Pause every 30 chunks
                    if (i + 1) % 30 == 0:
                        print("Taking a 60-second break in the summaries of chunks section...")
                        time.sleep(60)  # Sleep for 60 seconds (1 minute)

                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": tailored_prompt}],
                        model=model_input,
                        temperature=0.8,
                        max_tokens=1500,
                        top_p=1
                    )
                    summaries.append(response.choices[0].message.content)

                # Combine summaries in 1500-word chunks
                combined_summaries = []
                current_chunk = ""
                for summary in summaries:
                    if len(current_chunk) + len(summary) > 1000:
                        combined_summaries.append(current_chunk)
                        current_chunk = ""
                    current_chunk += summary + " "
                combined_summaries.append(current_chunk)  # Add the last chunk


                # Summarize the combined summaries into a final response
                temp_final_summary = ""
                for i, combined_summary in enumerate(combined_summaries):

                    # Pause every 30 chunks
                    if (i + 1) % 30 == 0:
                        print("Taking a 60-second break in the combined summaries section...")
                        time.sleep(60)  # Sleep for 60 seconds (1 minute)

                    combined_summary_prompt = f"""
                            Please summarize the following text, aiming to capture the most important points and central ideas. The summary should be concise and easy to understand, while still providing a clear and accurate overview of the original text. \
                            
                            Do not output anything else than the summary. Do not write an explanation of your thought process/introduction and do not write a conclusion: ONLY the summary.\
                            
                            Do not start with 'Here is the summary:' or any starting/introduction of the sort. Just output the result of your analysis.\

                            Here's the transcript to analyze and summarize: {combined_summary}
                        """

                    temp_final_response = client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": combined_summary_prompt}
                        ],
                        model=model_input,
                        temperature=0.8,
                        max_tokens=32000,
                        top_p=1
                    )
                    temp_final_summary += temp_final_response.choices[0].message.content + " "

                final_summary_prompt = f"""
                        You will receive summaries extracted from segments of a complete transcript. \
                        Your task is to synthesize these summaries into a comprehensive overview that captures the core message, key arguments, and essential details the author intended to convey. \
                        Focus on maintaining the author's voice and perspective while presenting a cohesive narrative of the transcript's content. \

                        Here's the transcript to analyze and summarize: {temp_final_summary}\

                        {question_to_answer}\
                """
            
                final_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": final_summary_prompt}],
                    model=model_input,
                    temperature=0.8,
                    max_tokens=32000,
                    top_p=1
                )

                return final_response.choices[0].message.content, summaries, selected_numbers

            final_summary, summaries_list, is_chunks_detailed = text_summarizer_agent(model_selected)

            print(final_summary)
            print("\n\n")

            if len(split_docs) > 20:
                key_takeaways_prompt = f"""
                    Analyze the following AI-generated summary of a transcript and create a concise and informative distillation for the reader. \

                    Please provide: \

                    1. A Compelling Headline:  A brief, attention-grabbing title that encapsulates the core message of the transcript. \
                    2. An Executive Summary: A concise paragraph (2-3 sentences) that summarizes the most critical points and insights from the transcript. \
                    3. Key Takeaways: A bulleted list of the 3-5 or 10 most important takeaways that the reader should remember and understand. Ensure these takeaways are actionable or provide a new perspective. \

                    Here is the AI-generated summary: \

                    {final_summary}
                """

                notes_taker_prompt = f"""
                    From this AI-generated summary of a full transcript, extract in a very thoughtful way all the information someone in the audience of this speaker would take while listening.\

                    Output your notes in a bullet points notes form. \

                    Here is the text to extract notes: \

                    {summaries_list}
                """
            else:
                key_takeaways_prompt = f"""
                    Analyze all the following summaries that have been AI-generated from a bunch of chunks derived from a full transcript and create a concise and informative distillation for the reader. \

                    Please provide: \

                    1. A Compelling Headline:  A brief, attention-grabbing title that encapsulates the core message of all the chunks summaries. \
                    2. An Executive Summary: A concise paragraph (2-3 sentences) that summarizes the most critical points and insights from all the chunks summaries \
                    3. Key Takeaways: A bulleted list of the 3-5 or 10 most important takeaways that the reader should remember and understand. Ensure these takeaways are actionable or provide a new perspective. \

                    Here is the AI-generated chunks summaries: \

                    {summaries_list}
                """

                notes_taker_prompt = f"""
                    From this list of pre-summurized chunks of text, that where generated from the full transcript of a presentation \
                    extract in a very thoughtful way all the information someone in the audience of this speaker would take while listening.\

                    Output your notes in a bullet points notes form. \

                    Here is the text to extract notes: \

                    {summaries_list}
                """

            enhanced_summary_prompt = f"""
                Please enhance the following summary generated by an AI model. Your goal is to make the summary more informative and engaging for the reader.\
                Instructions:\

                1) Maintain the Existing Structure: Keep the overall organization and sentence structure of the AI-generated summary.\
                2) Do Not Alter Original Sentences: Do not change any of the sentences directly provided in the summary.\
                3) Add Depth and Context:\
                    a) Clarify: Explain any ambiguous terms or concepts within the existing sentences.\
                    b) Elaborate: Expand upon key points with additional details and examples.\
                    c) Provide Context: Offer relevant background information to help the reader understand the significance of events or characters.\
                    d) Enhance Flow: Ensure the added content smoothly integrates with the original sentences.\

                Remember, do not alther the original sentences and do not make up facts, context or change the nature, tone or written style of the summary. \

                Here is the AI-generated summary: \

                {final_summary}
            """


            key_takeaways_response = client.chat.completions.create(
                messages=[{"role": "user", "content": key_takeaways_prompt}],
                model=model_selected,
                temperature=0.8,
                max_tokens=32000,
                top_p=1
            )

            notes_taker_response = client.chat.completions.create(
                messages=[{"role": "user", "content": notes_taker_prompt}],
                model=model_selected,
                temperature=0.8,
                max_tokens=32000,
                top_p=1
            )


            enhanced_response = client.chat.completions.create(
                messages=[{"role": "user", "content": enhanced_summary_prompt}],
                model=model_selected,
                temperature=0.8,
                max_tokens=32000,
                top_p=1
            )

            st.subheader("Final Summary")
            st.write(final_summary)
            
            st.subheader("Enhanced Summary")
            st.write(enhanced_response.choices[0].message.content)

            st.subheader("Key Takeaways")
            st.write(key_takeaways_response.choices[0].message.content)

            st.subheader("Notes")
            st.write(notes_taker_response.choices[0].message.content)
    else:
        st.warning("Please paste your transcript first.")
