import openai
import json

def generate_frame_descriptions(context, custom = True):
    if custom:
        openai.api_key = ""
        start_chat_log = f"""
        The following is a detailed and vivid description for a video, split into six distinct frames:

        Context: Flower budding
        Output:
        ["A group of closed buds can be seen on the stem of a plant.",
        "The buds begin to slowly unfurl, revealing small petals.",
        "The petals continue to unfurl, revealing more of the flower's center.",
        "The petals are now fully opened, and the center of the flower is visible.",
        "The flower's stamen and pistil become more prominent, and the petals start to curve outward.",
        "The fully bloomed flowers are in full view, with their petals open wide and displaying their vibrant colors."]

        Context: Volcanic eruption
        Output:
        ["A towering volcano stands against a backdrop of clear blue skies, with no visible signs of activity.",
         "Suddenly, a plume of thick smoke and ash erupts from the volcano's summit, billowing high into the air.",
        "Molten lava begins to flow down the volcano's slopes, glowing brightly with intense heat and leaving a trail of destruction.",
        "Explosions rock the volcano as fiery projectiles shoot into the sky, scattering debris and ash in all directions.",
        "The eruption intensifies, with a massive column of smoke and ash ascending into the atmosphere, darkening the surrounding area.",
       " As the eruption reaches its peak, a pyroclastic flow cascades down the volcano's sides, engulfing everything in its path with hot gases, ash, and volcanic material."]

        "Please generate a frame-by-frame description using a consistent and detailed narrative style as provided above for the given context. 
        The purpose of these frames is to seamlessly create a coherent video storyline. Ensure minimal changes between frames to maintain narrative continuity.
        Ensure the number of frames stays between 6 and 12. The output should be a Python list"

        Context: {context}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": start_chat_log}
            ],
            max_tokens=500
        )

        if isinstance(response.choices[0].message["content"], list):
            return response.choices[0].message["content"]
        else:
            return json.loads(response.choices[0].message["content"])
    else:
        import random
        list1 = ["A towering volcano stands against a backdrop of clear blue skies, with no visible signs of activity.",
         "Suddenly, a plume of thick smoke and ash erupts from the volcano's summit, billowing high into the air.",
        "Molten lava begins to flow down the volcano's slopes, glowing brightly with intense heat and leaving a trail of destruction.",
        "Explosions rock the volcano as fiery projectiles shoot into the sky, scattering debris and ash in all directions.",
        "The eruption intensifies, with a massive column of smoke and ash ascending into the atmosphere, darkening the surrounding area.",
       " As the eruption reaches its peak, a pyroclastic flow cascades down the volcano's sides, engulfing everything in its path with hot gases, ash, and volcanic material."]
        list2 = ["A group of closed buds can be seen on the stem of a plant.",
        "The buds begin to slowly unfurl, revealing small petals.",
        "The petals continue to unfurl, revealing more of the flower's center.",
        "The petals are now fully opened, and the center of the flower is visible.",
        "The flower's stamen and pistil become more prominent, and the petals start to curve outward.",
        "The fully bloomed flowers are in full view, with their petals open wide and displaying their vibrant colors."]
        
        list3 =  ["The sky, partially cloudy, with faint hints of colors starting to emerge.",
                  "A faint arch of colors becomes visible, stretching across the sky.",
                  "The rainbow gains intensity as the colors become more vibrant and defined.",
                  "The rainbow is now fully formed, displaying its classic arc shape.",
                  "The colors of the rainbow shine brilliantly against the backdrop of the sky.",
                  "The rainbow remains steady, its colors vivid and captivating as the rainbow decorates the sky."]
        lists = [list1, list2,list3]
        return lists[random.randint(0, len(lists))] #random.choice(lists)
