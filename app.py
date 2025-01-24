from typing import List


class Scene:
    index: int
    imagePrompt: str
    audioText: str
    imageFile: str | None
    audioFile: str | None


class Project:
    name: str
    story: str
    scenes: List[Scene]


def create_scenes():
    pass


def create_images():
    pass


def create_audio():
    pass


def image_to_video_with_captions_overlay_and_audio():
    pass


def merge_video_scenes():
    pass


def main():
    pass


'''
    get story name and story from user
    create scenes using ai
    create images using ai
    create audio using gtts
    merge images, audio and captions to single scene 
    merge all scenes to single video
'''

if __name__ == "__main__":
    main()
