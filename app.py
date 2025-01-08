import streamlit as st
import time
from typing import Optional, Dict, Set
from educhain import Educhain, LLMConfig
from langchain_openai import ChatOpenAI

# Improved templates with better formatting and type hints
INITIAL_QUESTION_TEMPLATE: str = """
Generate a unique and high-quality multiple-choice question (MCQ) based on the given topic and level.
The question should be clear, relevant, and aligned with the topic. Provide four answer options and the correct answer.

Topic: {topic}
Learning Objective: {learning_objective}
Difficulty Level: {difficulty_level}

Guidelines:
1. Avoid repeating questions.
2. Ensure the question is specific and tests knowledge effectively.
3. Provide plausible distractors (incorrect options).
4. Include a brief explanation for the correct answer.
"""

ADAPTIVE_QUESTION_TEMPLATE: str = """
Based on the user's response to the previous question on {topic}, generate a new unique and high-quality multiple-choice question (MCQ).
If the user's response is correct, output a harder question. Otherwise, output an easier question.
The question should be clear, relevant, and aligned with the topic. Provide four answer options and the correct answer.

Previous Question: {previous_question}
User's Response: {user_response}
Was the response correct?: {response_correct}

Guidelines:
1. Avoid repeating questions.
2. Ensure the question is specific and tests knowledge effectively.
3. Provide plausible distractors (incorrect options).
4. Include a brief explanation for the correct answer.
"""

@st.cache_resource
def get_llm(api_key: str) -> ChatOpenAI:
    """Initialize and cache the LLM client.
    
    Args:
        api_key: The API key for authentication
        
    Returns:
        ChatOpenAI: Initialized LLM client
    """
    return ChatOpenAI(
        model="llama-3.1-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key
    )

def generate_initial_question(topic: str, client: Educhain) -> Optional[Dict]:
    """Generate the first question for the quiz.
    
    Args:
        topic: The quiz topic
        client: The Educhain client
        
    Returns:
        Optional[Dict]: Question data or None if generation fails
    """
    try:
        result = client.qna_engine.generate_questions(
            topic=topic,
            num=1,
            learning_objective=f"General knowledge of {topic}",
            difficulty_level="Medium",
            prompt_template=INITIAL_QUESTION_TEMPLATE,
        )
        return result.questions[0] if result and result.questions else None
    except Exception as e:
        st.error(f"Failed to generate initial question: {str(e)}")
        return None

def generate_next_question(
    previous_question: str,
    user_response: str,
    response_correct: str,
    topic: str,
    client: Educhain,
    asked_questions: Set[str]
) -> Optional[Dict]:
    """Generate the next adaptive question based on user's performance.
    
    Args:
        previous_question: The last question asked
        user_response: User's answer to the previous question
        response_correct: Whether the user's answer was correct
        topic: The quiz topic
        client: The Educhain client
        asked_questions: Set of previously asked questions
        
    Returns:
        Optional[Dict]: Question data or None if generation fails
    """
    try:
        result = client.qna_engine.generate_questions(
            topic=topic,
            num=1,
            learning_objective=f"General knowledge of {topic}",
            difficulty_level="Harder" if response_correct == "True" else "Easier",
            prompt_template=ADAPTIVE_QUESTION_TEMPLATE,
            previous_question=previous_question,
            user_response=user_response,
            response_correct=response_correct
        )
        if result and result.questions:
            new_question = result.questions[0]
            if new_question.question not in asked_questions:
                return new_question
        return None
    except Exception as e:
        st.error(f"Failed to generate next question: {str(e)}")
        return None

def initialize_session_state() -> None:
    """Initialize or reset the session state variables."""
    if 'question_number' not in st.session_state:
        st.session_state.question_number = 0
        st.session_state.score = 0
        st.session_state.current_question = None
        st.session_state.topic = None
        st.session_state.start_time = None
        st.session_state.total_time = 0
        st.session_state.responses = []
        st.session_state.asked_questions = set()

def display_welcome_screen() -> None:
    """Display the welcome screen with instructions."""
    st.markdown("""
    ## Welcome to the Fast Adaptive Quiz!
    
    This quiz app uses AI to generate personalized questions based on your chosen topic and adapts to your performance.
    
    To get started:
    1. :red[Enter your GROQ API Key in the sidebar.] ← Start here!
    2. Choose a topic you want to study.
    3. Answer 5 questions and see how you do!
    
    The quiz will adjust its difficulty based on your answers, providing a tailored learning experience.
    
    Ready to challenge yourself? Let's begin!
    """)
    st.info("Enhance your learning with AI-powered quizzes!")

def display_quiz_summary() -> None:
    """Display the quiz completion summary and statistics."""
    st.success("Quiz completed!")
    st.balloons()
    st.metric("Final Score", f"{st.session_state.score}/5")
    st.metric("Total Time", f"{st.session_state.total_time:.2f} seconds")

    st.subheader("Quiz Summary")
    for i, response in enumerate(st.session_state.responses, 1):
        st.write(f"**Question {i}:** {response['question']}")
        st.write(f"**Your Answer:** {response['user_answer']}")
        st.write(f"**Correct Answer:** {response['correct_answer']}")
        st.write("---")

def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Fast Adaptive Quiz", layout="wide")

    # Sidebar setup
    with st.sidebar:
        st.title("Settings")
        st.markdown("### :red[Enter your GROQ API Key below]")
        api_key = st.text_input("GROQ API Key:", type="password")
        st.markdown("Follow me on [X](https://x.com/satvikps)")

    initialize_session_state()

    if not api_key:
        display_welcome_screen()
        st.stop()

    llm = get_llm(api_key)
    client = Educhain(LLMConfig(custom_model=llm))

    if st.session_state.question_number == 0:
        st.markdown("""
        ## Instructions
        1. Enter a topic you want to be quizzed on. It can be any subject or area of interest.
        2. The quiz consists of 5 multiple-choice questions.
        3. The difficulty of each question adapts based on your previous answer.
        4. Try to answer all questions to the best of your ability.
        5. Your total time will be tracked, so try to be both accurate and quick!

        Good luck and enjoy learning!
        """)
        topic = st.text_input("Enter the topic you want to study:")
        if st.button("Start Quiz", key="start_button"):
            st.session_state.topic = topic
            st.session_state.current_question = generate_initial_question(topic, client)
            if st.session_state.current_question:
                st.session_state.asked_questions.add(st.session_state.current_question.question)
                st.session_state.question_number += 1
                st.session_state.start_time = time.time()
                st.rerun()

    elif st.session_state.question_number <= 5:
        if st.session_state.current_question:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"Question {st.session_state.question_number}")
                st.write(st.session_state.current_question.question)
                
                user_answer = st.radio(
                    "Choose your answer:", 
                    st.session_state.current_question.options, 
                    key=f"q{st.session_state.question_number}"
                )
                
                if st.button("Submit Answer", key=f"submit{st.session_state.question_number}"):
                    correct_answer = st.session_state.current_question.answer
                    is_correct = user_answer == correct_answer
                    
                    if is_correct:
                        st.success("Correct!")
                        st.session_state.score += 1
                        response_correct = "True"
                    else:
                        st.error(f"Incorrect. The correct answer was {correct_answer}.")
                        response_correct = "False"

                    st.session_state.responses.append({
                        "question": st.session_state.current_question.question,
                        "user_answer": user_answer,
                        "correct_answer": correct_answer
                    })

                    if st.session_state.current_question.explanation:
                        st.info(f"Explanation: {st.session_state.current_question.explanation}")

                    st.session_state.question_number += 1

                    if st.session_state.question_number <= 5:
                        next_question = generate_next_question(
                            st.session_state.current_question.question,
                            user_answer,
                            response_correct,
                            st.session_state.topic,
                            client,
                            st.session_state.asked_questions
                        )
                        if next_question:
                            st.session_state.current_question = next_question
                            st.session_state.asked_questions.add(next_question.question)
                        else:
                            st.error("Failed to generate a new question. Please try again.")
                            st.session_state.question_number = 6
                    else:
                        st.session_state.total_time = time.time() - st.session_state.start_time
                    st.rerun()
            
            with col2:
                st.metric("Score", f"{st.session_state.score}/{st.session_state.question_number - 1}")
                st.progress(st.session_state.question_number / 5)
                elapsed_time = time.time() - st.session_state.start_time
                st.metric("Time", f"{elapsed_time:.2f} seconds")

    if st.session_state.question_number > 5:
        display_quiz_summary()
        if st.button("Restart Quiz"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
