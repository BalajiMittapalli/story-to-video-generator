


import streamlit as st
import os
import time
from pathlib import Path
from app import (get_project, create_scenes, create_images,
                create_audio, create_scene_videos, create_final_scenes,
                merge_final_scenes, GENERATIONS_DIR, prompt_template)

def show_progress_step(label, func, project, *args):
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown("⏳")  # Loading icon
        
    with col2:
        st.markdown(f"**{label}**")
        progress_bar = st.progress(0)
        
        try:
            start_time = time.time()
            func(project, *args)
            elapsed = time.time() - start_time
            
            col1.empty()
            with col1:
                st.success("✓")
            with col2:
                st.markdown(f"`Completed in {elapsed:.1f}s`")
                
        except Exception as e:
            col1.empty()
            with col1:
                st.error("✗")
            with col2:
                st.error(f"Failed: {str(e)}")
            raise

def display_project_contents(project_path):
    with st.sidebar:
        st.subheader("Project Contents")
        for item in project_path.glob("*"):
            if item.is_dir():
                with st.expander(f"📁 {item.name}"):
                    for subitem in item.glob("*"):
                        st.write(f"📄 {subitem.name}")
            else:
                st.write(f"📄 {item.name}")

def display_project_assets(project_path):
    st.header("Project Assets")
    final_video = project_path / "final.mp4"
    if final_video.exists():
        st.subheader("Final Film")
        st.video(str(final_video))
    
    images_dir = project_path / "images"
    if images_dir.exists():
        st.subheader("Generated Images")
        cols = st.columns(3)
        for idx, img in enumerate(images_dir.glob("*.png")):
            cols[idx%3].image(str(img))
    
    videos_dir = project_path / "videos"
    if videos_dir.exists():
        st.subheader("Scene Videos")
        for vid in videos_dir.glob("*.mp4"):
            st.video(str(vid))

def display_existing_projects():
    st.sidebar.header("Existing Stories")
    projects = [d for d in Path(GENERATIONS_DIR).glob("*") if d.is_dir()]
    
    if not projects:
        st.sidebar.write("No existing stories found")
        return None
    
    selected = st.sidebar.selectbox("Choose a story", 
                                  [p.name for p in projects],
                                  index=None)
    if selected:
        selected_path = Path(GENERATIONS_DIR) / selected
        display_project_contents(selected_path)
        return selected_path
    return None

def new_story_form():
    with st.form("new_story_form"):
        st.header("Create New Story")
        name = st.text_input("Story Name")
        story = st.text_area("Story Content", height=200)
        submitted = st.form_submit_button("Generate Animation")
        
        if submitted:
            if not name or not story:
                st.error("Please fill in all fields")
                return None
            return get_project(name, story)
    return None

def main():
    st.set_page_config(page_title="Folk Story Animator", layout="wide")
    st.title("Folktale to Animation Converter")
    
    # Check for existing project selection
    existing_project_path = display_existing_projects()
    
    if existing_project_path:
        display_project_assets(existing_project_path)
    else:
        # Show new story form by default
        project = new_story_form()
        if project:
            st.session_state.project = project
            st.session_state.current_step = 0
    
    # Handle generation progress
    if 'project' in st.session_state:
        project = st.session_state.project
        progress_steps = [
            ("Generating Scenes", create_scenes, prompt_template),
            ("Creating Images", create_images),
            ("Generating Audio", create_audio),
            ("Rendering Scenes", create_scene_videos),
            ("Finalizing Scenes", create_final_scenes),
            ("Merging Film", merge_final_scenes)
        ]
        
        with st.container():
            st.subheader("Generation Progress")
            for i, (label, func, *args) in enumerate(progress_steps):
                if i < st.session_state.get('current_step', 0):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1: st.success("✓")
                    with col2: st.markdown(f"**{label}**")
                    continue
                
                if i == st.session_state.get('current_step', 0):
                    show_progress_step(label, func, project, *args)
                    st.session_state.current_step = i + 1
                
                if st.session_state.current_step >= len(progress_steps):
                    st.balloons()
                    st.success("Animation Complete!")
                    final_video = Path(project.project_dir) / "final.mp4"
                    if final_video.exists():
                        st.video(str(final_video))
                    break

if __name__ == "__main__":
    main()