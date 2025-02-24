import streamlit as st
import json
import os
from my_workflow import run_pipeline
st.markdown(
    """
    <style>
    /* Target the disabled text areas and force normal look. */
    textarea:disabled,
    textarea[disabled],
    textarea[disabled] *
      {
        background-color: #00000000 !important;  /* background color */
        color: #fafafa !important;             /* text color */
        -webkit-text-fill-color: #fafafa !important; /* Chrome-based browsers fix */
        cursor: text !important;               /* show normal text cursor */
        opacity: 1 !important;                 /* remove the “faded” look */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# A list of the possible query files we have:
AVAILABLE_QUERIES = ["query1.txt", "query2.txt", "query3.txt", "query4.txt"]

# A list of our two resource files that we allow editing
RESOURCE_FILES = ["grades.csv", "students.txt"]

def main():
    # Show the deployed commit hash for debugging
    if os.path.exists("version.txt"):
        with open("version.txt", "r", encoding="utf-8-sig") as f:
            commit_hash = f.read().strip()
        st.info(f"**Deployed commit:** {commit_hash}")

    if "file_contents" not in st.session_state:
        st.session_state["file_contents"] = {}

    st.title("Nir Elkayam's LangChain Demo")

    # ---- New Description for the user ----
    st.markdown(
        """
        **Welcome to my LangChain Demo!**

        This application demonstrates a **multi-step, agent-based system** powered by [LangChain](https://github.com/hwchase17/langchain). 
        Each *query* file (e.g., `query1.txt`) contains **instructions** or a **prompt** that the pipeline will interpret. 
        Behind the scenes, the system uses an **LLM** and other tools to **plan** which "tool" or function to call next, it can, for example:
        - Generate Python code to analyze a CSV file, 
        - Execute that code and capture the result, 
        - Perform text analysis on files, 
        - Search the web for information using duckduckgo api,
        - Finalize an answer once it has all the information it needs.

        **How to use this app**:
        1. **Pick** one of the available query files from the dropdown.
        2. **View or edit** the query file’s text in the text area (press "Save Query File" to confirm changes).
        3. **Review** the input JSON and other resource files for context.
        4. Click **"Run Pipeline"** to see how the LLM decides to handle the query. The logs and final state are shown in a collapsible 
           section, and any newly created or modified files will appear below.
        5. Because we dynamically generate new code on the fly, the pipeline may not always produce a perfect result.
           Feel free to click "Run Pipeline" again if the output is unexpected.

        *Note: All edits and new files occur only within your current session, so you can safely experiment!*
         """,
        unsafe_allow_html=True
    )


    # ---------------------------
    # 1) Pick and Edit Query File
    # ---------------------------

    selected_query = st.selectbox("Pick a query file to load:", AVAILABLE_QUERIES)
    
    default_query_content = ""
    if os.path.exists(selected_query):
        with open(selected_query, "r", encoding="utf-8") as f:
            default_query_content = f.read()
    else:
        default_query_content = f"{selected_query} not found."

    edited_query_content = st.text_area(
        label=f"Edit the content of {selected_query}:",
        value=default_query_content,
        height=200
    )

    if st.button("Save Query File"):
        st.session_state["file_contents"][selected_query] = edited_query_content
        st.success(f"Saved changes to {selected_query} (session-only).")

    # ------------------------------
    # 2) Default JSON (input.json)
    # ------------------------------
    default_json = {
        "query_name": selected_query,
        "file_resources": [
            {
                "file_name": "grades.csv",
                "description": (
                    "A csv file describing students and their grades. "
                    "The first line contains the columns (field names) of each following row. "
                    "They are: First, Last, University-Year, Class, Grade. "
                    "An example row is: Shalom, Levi, 3, Calculus, 93.5"
                )
            },
            {
                "file_name": "students.txt",
                "description": (
                    "A text file containing information on some of the students, "
                    "including his or her aspirations, favorite city, and hobbies."
                )
            }
        ]
    }

    default_json_str = json.dumps(default_json, indent=4)
    user_json_str = st.text_area(
        "View input JSON:",
        value=default_json_str,
        height=200,
        disabled=True
    )

    # --------------------------------------------
    # 3) Let the user view/edit resource files too
    # --------------------------------------------

    st.subheader("Resource Files")
    st.write(
        "Here you can view the content of `grades.csv` or `students.txt`."
    )

    selected_resource = st.selectbox("Pick a resource file to view:", RESOURCE_FILES)

    default_resource_content = ""
    # First check if we have a session-based edit saved
    if selected_resource in st.session_state["file_contents"]:
        default_resource_content = st.session_state["file_contents"][selected_resource]
    else:
        # Otherwise, read from disk as a fallback
        if os.path.exists(selected_resource):
            with open(selected_resource, "r", encoding="utf-8") as f:
                default_resource_content = f.read()
        else:
            default_resource_content = f"{selected_resource} not found."
        



    
    edited_resource_content = st.text_area(
        label=f"View the content of {selected_resource}:",
        value=default_resource_content,
        height=200,
        disabled=True
    )

    # if st.button("Save Resource File"):
    #     st.session_state["file_contents"][selected_resource] = edited_resource_content
    #     st.success(f"Saved changes to {selected_resource} (session-only).")

    # --------------------------------------------
    # 4) Run the pipeline


    if st.button("Run Pipeline"):
        try:
            input_data = json.loads(user_json_str)
            if selected_query in st.session_state["file_contents"]:
                input_data["query_content"] = st.session_state["file_contents"][selected_query]
            # Inject any session-edited files into input_data
            if "file_contents" in st.session_state and "file_resources" in input_data:
                for resource in input_data["file_resources"]:
                    fname = resource.get("file_name") or resource.get("name")
                    if fname in st.session_state["file_contents"]:
                        resource["content"] = st.session_state["file_contents"][fname]
            
            # Run the pipeline with our updated input data
            final_state = run_pipeline(input_data)

            with st.expander("Click to see Logs / Final State", expanded=False):
                st.subheader("Final State")
                st.json(final_state)
                st.write("**created_files** =", final_state.get("created_files", "No created_files in final_state"))

                if "program_output" in final_state:
                    st.subheader("Program Output")
                    st.text(final_state["program_output"])

            st.subheader("Created / Modified Files")
            if "created_files" in final_state:
                file_list = final_state["created_files"]
                unique_files = []
                for fname in file_list:
                    if fname not in unique_files:
                        unique_files.append(fname)
                for filename in unique_files:
                    display_file_contents(filename)
            else:
                st.write("No 'created_files' key found in final_state.")

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")


        

def display_file_contents(filename: str):
    if os.path.exists(filename):
        st.markdown(f"**{filename}**")
        if filename.endswith(".py"):
            with open(filename, "r") as f:
                code = f.read()
            st.code(code, language="python")
            st.download_button(
                label=f"Download {filename}",
                data=code,
                file_name=filename,
                mime="text/x-python",
                key=f"download_{filename}"
            )

        elif filename.endswith(".json"):
            with open(filename, "r") as f:
                data = f.read()
            st.json(data)
            st.download_button(
                label=f"Download {filename}",
                data=data,
                file_name=filename,
                mime="application/json",
                key=f"download_{filename}"
            )

        elif filename.endswith(".txt") or filename.endswith(".csv"):
            with open(filename, "r") as f:
                text = f.read()
            st.text(text)
            st.download_button(
                label=f"Download {filename}",
                data=text,
                file_name=filename,
                mime="text/plain",
                key=f"download_{filename}"
            )
        else:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            st.text(content)
    else:
        st.warning(f"File not found: {filename}")


if __name__ == "__main__":
    main()
