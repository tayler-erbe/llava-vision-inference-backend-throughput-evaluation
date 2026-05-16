# Prompt Shape Experiment — 1dc91d5d-897a-49b5-b1b0-4c6db1f0f6b0

- Date: 2026-05-13T20:44:00.107341+00:00
- Host: analytics-server.example.local
- Images: 50  |  Repeats per cell: 3  |  Warmup excluded: 2 calls/variant

## Variant overview

| Variant | Calls per image | What it asks for |
|---|---|---|
| A | 3 | natural desc + strict desc + moderation (current pipeline) |
| B | 4 | title + natural desc + strict desc + moderation |
| C | 1 | combined JSON: title + description + offensive + category + reason + confidence |

## Latency & throughput (per-image, aggregated across all repeats)

| Variant | Cells measured | Wall median (s/img) | Wall mean (s/img) | Wall std (s) | Wall min/max | Total wall (s) | Tokens/sec |
|---|---|---|---|---|---|---|---|
| A | 150 | 10.78 | 11.05 | 2.79 | 4.03/16.43 | 1657.8 | 11.1 |
| B | 150 | 11.96 | 12.05 | 2.73 | 6.05/17.24 | 1807.2 | 11.1 |
| C | 148 | 6.70 | 6.86 | 1.48 | 4.54/12.09 | 1014.8 | 11.8 |

**Combined-prompt speedup vs current 3-prompt: 1.61x** (based on per-image wall mean)

## Per-field parse rate

`valid` means: field present AND passed the schema check (correct type, length, allowed value set, etc).

| Variant | description | offensive | offense_category | offense_reason | title | confidence |
|---|---|---|---|---|---|---|
| A | 100% | 100% | 100% | 100% | n/a | n/a |
| B | 100% | 100% | 100% | 100% | 100% | n/a |
| C | 100% | 100% | 100% | 100% | 94% | 100% |

## Repeat consistency (temperature=0, 3 repeats)

At temperature=0, the model SHOULD produce identical outputs across repeats of the same (image, prompt) cell. Any divergence indicates nondeterminism in the inference engine — useful to know for production stability.

| Variant | Cells | Identical across repeats | Divergent | Identical % |
|---|---|---|---|---|
| A | 150 | 150 | 0 | 100% |
| B | 200 | 200 | 0 | 100% |
| C | 50 | 50 | 0 | 100% |

## Per-image content spot-check (repeat 0 only)

Side-by-side outputs for each image, so you can eyeball whether the combined prompt's description is comparably useful to the separate prompts. Truncated to first 200 chars per field. Only the first repeat is shown here; see `outputs.jsonl` for the rest.

### 0000001.jpg

**Variant A:**
- `description`: The image is a black and white photograph of an empty stadium. The stadium has a large roof supported by several pillars, and there are scaffolding on the side of the building. The ground around the s…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of an empty stadium with a building under construction in the background. It does not contain any identifiable sensitive categories.

**Variant B:**
- `description`: The image is a black and white photograph of an empty stadium. The stadium has a large roof supported by several pillars, and there are scaffolding on the side of the building. The ground around the s…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of an empty stadium with a building under construction in the background. It does not contain any identifiable sensitive categories.
- `title`: An old stadium under construction.

**Variant C:**
- `title`: A stadium under construction with workers on the roof
- `description`: The image shows a large building in the process of being built. Workers can be seen on the roof, and scaffolding is visible around the structure.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000002.jpg

**Variant A:**
- `description`: A black and white drawing of a football stadium with the words "Championship of Illinois" written at the top.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white illustration of a football stadium with players on the field. It does not contain any offensive content.

**Variant B:**
- `description`: A black and white drawing of a football stadium with the words "Championship of Illinois" written at the top.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white illustration of a football stadium with players on the field. It does not contain any offensive content.
- `title`: A black and white drawing of a football stadium.

**Variant C:**
- `title`: Archives of Illinois
- `description`: A black and white photograph of a football stadium with the words Archives of Illinois written on it.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: low

### 0000003.jpg

**Variant A:**
- `description`: The image is a black and white photograph of the exterior of the University of Texas at Austin. The building, which is the main focus of the image, has a large stadium-like structure on top. This stru…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of the exterior of the University of Texas at Austin. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of the exterior of the University of Texas at Austin. The building, which is the main focus of the image, has a large stadium-like structure on top. This stru…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of the exterior of the University of Texas at Austin. It does not contain any offensive content.
- `title`: A black and white photo of a stadium with the words "University of Texas" on it.

**Variant C:**
- `title`: Aerial view of a stadium with a large crowd in the stands and a scoreboard on top
- `description`: The image shows an aerial view of a stadium filled with people. The scoreboard is prominently displayed at the top of the building.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000004.jpg

**Variant A:**
- `description`: The image is a black and white poster of Pres. James. It features four photographs of him, one of which shows him on horseback. The poster also includes text that reads "Illinois" and "Pres. James".
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white poster of Pres. James. It features four photographs of him, one of which shows him on horseback. The poster also includes text that reads "Illinois" and "Pres. James".
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: An old photo of Pres. James

**Variant C:**
- `title`: Pres. James
- `description`: A black and white photograph of President James with a horse in the foreground.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000005.jpg

**Variant A:**
- `description`: The image is a black and white photo of four men standing around a pig. The man on the left is holding a small object in his hand, while the man on the right is holding a stick. The pig is standing on…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photo of four men standing around a pig. The man on the left is holding a small object in his hand, while the man on the right is holding a stick. The pig is standing on…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Men looking at a pig

**Variant C:**
- `title`: Three men looking at a pig
- `description`: The image shows three men in suits and hats looking at a large pig. The pig is standing on its hind legs, appearing to be performing tricks for the men.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000006.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a rural landscape. The main focus is a field, which is divided into two sections by a fence. On the left side of the field, there are several trees scatter…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a field with houses in the background. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of a rural landscape. The main focus is a field, which is divided into two sections by a fence. On the left side of the field, there are several trees scatter…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a field with houses in the background. It does not contain any offensive content.
- `title`: A black and white photo of a farm with cows.

**Variant C:**
- `title`: A black and white photograph of a rural area with houses and trees
- `description`: The image is in black and white and shows a rural area with several small houses and trees. The photo appears to be old, possibly from the early 20th century.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000007.jpg

**Variant A:**
- `description`: This is a black and white photograph capturing a moment in the life of a herd of cattle. The image features six cows, each exhibiting different shades of gray, suggesting varying ages or stages of gro…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of cows in a field. It does not contain any offensive content.

**Variant B:**
- `description`: This is a black and white photograph capturing a moment in the life of a herd of cattle. The image features six cows, each exhibiting different shades of gray, suggesting varying ages or stages of gro…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of cows in a field. It does not contain any offensive content.
- `title`: A herd of cattle in a field.

**Variant C:**
- `title`: A herd of cows in a field
- `description`: The image shows several cows grazing in a large open field. The cows are scattered throughout the field, some standing and others lying down. The field is surrounded by a fence, indicating that it's l…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000008.jpg

**Variant A:**
- `description`: The image is a black and white photograph capturing a moment in time at a farm. The main subjects of the image are two men and a group of cows.

One man, dressed in overalls, stands next to a cow that…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of cows in a pen with a man standing next to them. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph capturing a moment in time at a farm. The main subjects of the image are two men and a group of cows.

One man, dressed in overalls, stands next to a cow that…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of cows in a pen with a man standing next to them. It does not contain any offensive content.
- `title`: Cowboys in a pen with cattle

**Variant C:**
- `title`: A man in a cowboy hat standing next to cows
- `description`: The image shows a man wearing a cowboy hat and standing next to several cows. The cows are black and white, and they appear to be in a pen or corral. There is also a train visible in the background.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000009.jpg

**Variant A:**
- `description`: This is a black and white photograph capturing a serene rural scene. The main subjects of the image are two cows, one black and one brown, standing in a field. They appear to be grazing peacefully on …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of cows grazing in a field. It does not contain any offensive content.

**Variant B:**
- `description`: This is a black and white photograph capturing a serene rural scene. The main subjects of the image are two cows, one black and one brown, standing in a field. They appear to be grazing peacefully on …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of cows grazing in a field. It does not contain any offensive content.
- `title`: Three cows in a field.

**Variant C:**
- `title`: A black and white photograph of cows in a field
- `description`: The image is a black and white photograph capturing a moment in the life of cows. The cows are grazing on dry grass, their bodies standing out against the monochrome backdrop. In the distance, a build…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000010.jpg

**Variant A:**
- `description`: The image is a black and white photograph of an older woman. She is wearing glasses, a striped shirt, and has her arms crossed. The background consists of a bookshelf filled with books and papers scat…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of an older woman. She is wearing glasses, a striped shirt, and has her arms crossed. The background consists of a bookshelf filled with books and papers scat…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Woman in striped shirt and glasses.

**Variant C:**
- `title`: Woman in striped shirt and glasses
- `description`: A woman wearing a striped shirt and glasses is sitting at a desk. She has her arms crossed.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000011.jpg

**Variant A:**
- `description`: Two men in suits are shaking hands. The man on the left is wearing a black suit with a striped tie, while the man on the right is wearing a gray suit with a polka dot tie. They are standing against a …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: Two men in suits are shaking hands. The man on the left is wearing a black suit with a striped tie, while the man on the right is wearing a gray suit with a polka dot tie. They are standing against a …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Two men shake hands.

**Variant C:**
- `title`: Two men shaking hands
- `description`: A black and white photograph of two men in suits shaking hands. One man is wearing a tie with stripes, while the other has a solid color tie.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000012.jpg

**Variant A:**
- `description`: The image is a black and white photo of four men standing in front of an American flag. The man on the far left is wearing a suit and tie, while the other three are dressed more casually. The man in t…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photo of four men standing in front of an American flag. The man on the far left is wearing a suit and tie, while the other three are dressed more casually. The man in t…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Four men in suits and one man in a white suit.

**Variant C:**
- `title`: Four men in suits and ties standing together
- `description`: The four men are wearing suits and ties. They appear to be posing for a photo.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000013.jpg

**Variant A:**
- `description`: A group of men in suits are posing for a photo with one man holding up a plaque.
- `offensive`: False
- `offense_category`: Other sensitive categories
- `offense_reason`: The image contains a group of men posing for a photo with one man holding up a plaque. The image does not contain any offensive content.

**Variant B:**
- `description`: A group of men in suits are posing for a photo with one man holding up a plaque.
- `offensive`: False
- `offense_category`: Other sensitive categories
- `offense_reason`: The image contains a group of men posing for a photo with one man holding up a plaque. The image does not contain any offensive content.
- `title`: A group of men posing for a picture with one man holding up a plaque.

**Variant C:**
- `title`: Group of people holding a framed certificate
- `description`: A group of men and women are standing together, holding a framed certificate. The certificate is white with black text and a gold seal in the center.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000014.jpg

**Variant A:**
- `description`: A group of people posing for a photo. The man on the far left is wearing a suit and tie, while the woman next to him has a scarf around her neck. The third person from the left is holding a stuffed an…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: A group of people posing for a photo. The man on the far left is wearing a suit and tie, while the woman next to him has a scarf around her neck. The third person from the left is holding a stuffed an…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A group of people posing for a photo.

**Variant C:**
- `title`: A group of people posing for a photo
- `description`: The image shows a group of people standing together, smiling and posing for the camera. They are dressed in formal attire, suggesting that this might be a special occasion or event.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000015.jpg

**Variant A:**
- `description`: Two men shake hands in front of a curtain. One man is wearing a suit and tie, while the other is dressed in a jacket and pants. They are standing in front of a table that has a microphone on it. The b…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: Two men shake hands in front of a curtain. One man is wearing a suit and tie, while the other is dressed in a jacket and pants. They are standing in front of a table that has a microphone on it. The b…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Two men shake hands in front of a curtain.

**Variant C:**
- `title`: Two men shaking hands in front of a curtain with a swastika on it.
- `description`: The image shows two men shaking hands in front of a curtain that has a swastika on it. The men are standing in front of a table that has some papers and a microphone on it.
- `offensive`: True
- `offense_category`: Hate symbols
- `offense_reason`: A swastika is visible on the curtain behind the two men.
- `confidence`: high

### 0000016.jpg

**Variant A:**
- `description`: The image is a black and white portrait of an older man. He has a mustache and is wearing a suit with a bow tie. The photo was taken by Ashley Young Photography.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white portrait of an older man. He has a mustache and is wearing a suit with a bow tie. The photo was taken by Ashley Young Photography.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man with a mustache and white hair.

**Variant C:**
- `title`: Portrait of an older man with a mustache and bow tie
- `description`: A black and white photograph of a man wearing a suit and tie. The image is in portrait orientation.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000017.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a man. He has a mustache and goatee, giving him an air of sophistication. His attire consists of a suit jacket with a tie, suggesting a formal occasion or …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of a man. He has a mustache and goatee, giving him an air of sophistication. His attire consists of a suit jacket with a tie, suggesting a formal occasion or …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man in a suit and tie.

**Variant C:**
- `title`: Man in suit and tie
- `description`: A man wearing a suit and tie. He is holding a briefcase in his hand.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000018.jpg

**Variant A:**
- `description`: The image is a black and white photograph of an older man. He has a mustache and is wearing a suit with three buttons on the front. The man's gaze is directed towards the camera, giving the impression…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of an older man. He has a mustache and is wearing a suit with three buttons on the front. The man's gaze is directed towards the camera, giving the impression…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man with a mustache and tie.

**Variant C:**
- `title`: A man in a suit and tie with a mustache
- `description`: The image is black and white.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000019.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a large brick building with arched windows. The building has a sign that reads "G. W. 16-17". There are two people walking in front of the building, one on…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of a large brick building with arched windows. The building has a sign that reads "G. W. 16-17". There are two people walking in front of the building, one on…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A black and white photo of a building with trees in front.

**Variant C:**
- `title`: A black and white photograph of a brick building with arched windows and a street scene in front of it.
- `description`: The image is a black and white photograph capturing a moment on a city street. The main focus is a large, two-story brick building with arched windows. The building's architecture suggests it could be…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000020.jpg

**Variant A:**
- `description`: The image features a statue of a man, standing on a pedestal. The statue is made of stone and has a long robe draped over its body. It is located in a park, surrounded by trees. The photo was taken du…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image features a statue of a man, standing on a pedestal. The statue is made of stone and has a long robe draped over its body. It is located in a park, surrounded by trees. The photo was taken du…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Statue of a man in front of trees.

**Variant C:**
- `title`: Statue of a person in a robe
- `description`: A statue of a person in a robe is standing on a pedestal. The statue is located near trees and a body of water.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000021.jpg

**Variant A:**
- `description`: The image captures a large indoor stadium, viewed from an elevated perspective. The floor of the stadium is covered in artificial turf, and there are rows upon rows of empty seats surrounding it. A ba…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image captures a large indoor stadium, viewed from an elevated perspective. The floor of the stadium is covered in artificial turf, and there are rows upon rows of empty seats surrounding it. A ba…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A black and white photo of a basketball court.

**Variant C:**
- `title`: A large stadium with a basketball court in the center
- `description`: The image is black and white and shows an empty stadium with a basketball court in the middle. The court has a net at one end and a hoop at the other.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000022.jpg

**Variant A:**
- `description`: The image is a black and white photo of a large stadium. The stadium has a dome roof with a circular pattern on it. There are multiple parking lots surrounding the stadium, filled with cars. The photo…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a large stadium with a dome shaped roof. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photo of a large stadium. The stadium has a dome roof with a circular pattern on it. There are multiple parking lots surrounding the stadium, filled with cars. The photo…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a large stadium with a dome shaped roof. It does not contain any offensive content.
- `title`: Aerial view of a stadium with cars and buses in the parking lot.

**Variant C:**
- `title`: A large white dome shaped building with a circular opening at the top and a smaller circular opening on the side. The building is surrounded by parking lots filled with cars.
- `description`: The image shows a large, white dome-shaped building with a circular opening at the top and a smaller circular opening on the side. The building is surrounded by multiple parking lots filled with cars.…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000023.jpg

**Variant A:**
- `description`: The image is a black and white photo of an Indian tribe performing on a field. The tribe members are dressed in traditional clothing, with one member holding a drum. They are standing on the grass, wh…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photo of an Indian tribe performing on a field. The tribe members are dressed in traditional clothing, with one member holding a drum. They are standing on the grass, wh…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Indian man with a feathered headdress and a white shirt.

**Variant C:**
- `title`: Native American Indian Dance
- `description`: A group of Native Americans performing a traditional dance in front of an audience. The dancers are adorned with colorful headdresses and clothing, adding to the vibrancy of the scene.
- `offensive`: False
- `offense_category`: Native American imagery
- `offense_reason`: —
- `confidence`: high

### 0000024.jpg

**Variant A:**
- `description`: The image captures a moment of triumph for the band, as they stand on the field with their instruments. The band members are dressed in black uniforms, creating a stark contrast against the white back…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image captures a moment of triumph for the band, as they stand on the field with their instruments. The band members are dressed in black uniforms, creating a stark contrast against the white back…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A band performing on a field with an eagle propeller.

**Variant C:**
- `title`: A band performing in front of a crowd
- `description`: The band is playing on a field with the audience watching. The photo captures the moment from the side of the stage.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000025.jpg

**Variant A:**
- `description`: The image is a black and white photo of an Indian man. He is wearing a headdress, a shirt with fringe on the sleeves, and pants. The man is standing in a field with trees in the background. He is hold…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photo of an Indian man. He is wearing a headdress, a shirt with fringe on the sleeves, and pants. The man is standing in a field with trees in the background. He is hold…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Indian man in a field with his hand up.

**Variant C:**
- `title`: Native American man in headdress and traditional clothing with arm raised
- `description`: A black and white photograph of a Native American man wearing a headdress and traditional clothing, standing on a grassy field with trees in the background. He is holding an object in his hand and has…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000026.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a building. The building has a large arched entrance, above which hangs a sign that reads "Gentlemen's Club". To the left of the entrance, there are two tr…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a building with trees in front. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of a building. The building has a large arched entrance, above which hangs a sign that reads "Gentlemen's Club". To the left of the entrance, there are two tr…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a building with trees in front. It does not contain any offensive content.
- `title`: A black and white photo of a building with trees in front.

**Variant C:**
- `title`: A large brick building with a clock tower and a statue of a man in front of it.
- `description`: The building is made of bricks and has a clock tower. In front of the building, there is a statue of a man. The photo was taken during the day.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000027.jpg

**Variant A:**
- `description`: The image captures a lively scene at an indoor sporting event. The main focus is on the crowd of spectators, who are standing and sitting in rows, their attention riveted on the court below. They are …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image captures a lively scene at an indoor sporting event. The main focus is on the crowd of spectators, who are standing and sitting in rows, their attention riveted on the court below. They are …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A black and white photo of a basketball game

**Variant C:**
- `title`: A black and white photograph of a crowd at a sporting event
- `description`: The image captures a moment in time at a sporting event, with the crowd standing on their feet, cheering for their team. The photograph is in black and white, adding a timeless quality to the scene.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000028.jpg

**Variant A:**
- `description`: A woman wearing a Native American headdress and dress.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: A woman wearing a Native American headdress and dress.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Indian woman in a headdress and fringed skirt.

**Variant C:**
- `title`: Native American woman in traditional attire with a headdress and fringed skirt holding a pipe
- `description`: The image is black and white and features a Native American woman wearing traditional clothing, including a headdress and fringed skirt. She is holding a pipe in her hand.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000029.jpg

**Variant A:**
- `description`: A man stands in front of a poster featuring an Indian chief. The man is dressed in a suit and tie, while the Indian chief is depicted wearing a headdress and holding a pipe. The background features a …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: A man stands in front of a poster featuring an Indian chief. The man is dressed in a suit and tie, while the Indian chief is depicted wearing a headdress and holding a pipe. The background features a …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man standing in front of a poster of an indian.

**Variant C:**
- `title`: Native American man in headdress with hand raised
- `description`: A black and white photograph of a Native American man wearing a headdress and raising his hand. He is standing next to a man dressed in a suit. The background features a mural depicting a Native Ameri…
- `offensive`: True
- `offense_category`: Native American imagery
- `offense_reason`: The image contains a Native American person wearing a headdress, which may be considered offensive by some viewers.
- `confidence`: high

### 0000030.jpg

**Variant A:**
- `description`: The image is a black and white photograph featuring four men. The man on the far left is wearing a suit and tie, while the other three men are dressed in suits without ties. They are standing together…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of four men in suits. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph featuring four men. The man on the far left is wearing a suit and tie, while the other three men are dressed in suits without ties. They are standing together…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of four men in suits. It does not contain any offensive content.
- `title`: Ousted U.L. officials meet board

**Variant C:**
- `title`: Ousted U.L. officials meet board
- `description`: The meeting was attended by the following members: John W. O'Connell, James E. Johnson, and Thomas J. O'Sullivan.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: low

### 0000031.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a man. The man has a beard and is wearing a suit with a bow tie. He is looking to the left of the frame. The background is plain and there are no other obj…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of a man. The man has a beard and is wearing a suit with a bow tie. He is looking to the left of the frame. The background is plain and there are no other obj…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man with a beard and bow tie.

**Variant C:**
- `title`: D. Terrellon
- `description`: A man in a suit and tie is wearing a mustache.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: medium

### 0000032.jpg

**Variant A:**
- `description`: A man in a suit is sitting at a desk with papers and books. He has a pen in his hand, suggesting he might be writing or drawing. The room appears to be an office or study, as there are other desks vis…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: A man in a suit is sitting at a desk with papers and books. He has a pen in his hand, suggesting he might be writing or drawing. The room appears to be an office or study, as there are other desks vis…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man in a suit with a tie and glasses.

**Variant C:**
- `title`: A man in a suit writing at a desk
- `description`: The image shows a man in a suit sitting at a desk with papers and books around him. He is wearing glasses and appears to be focused on his work.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000033.jpg

**Variant A:**
- `description`: The image is a black and white photograph capturing a serene scene in front of a large building. The building, which is the main focus of the image, has a prominent tower on its left side. It features…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph capturing a serene scene in front of a large building. The building, which is the main focus of the image, has a prominent tower on its left side. It features…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image does not contain any offensive content.
- `title`: A black and white photo of a building with a truck in front.

**Variant C:**
- `title`: A black and white photograph of a building with a truck in front of it.
- `description`: The image is a black and white photograph capturing a moment in time. It features a large building, possibly an institution or government building, with a truck parked in front of it. The building has…
- `offensive`: False
- `offense_category`: Other sensitive categories
- `offense_reason`: —
- `confidence`: high

### 0000034.jpg

**Variant A:**
- `description`: The image captures a moment in time on a tree-lined street. A group of people are seen walking down the sidewalk, their shadows stretching out behind them. The photo is taken from a low angle, giving …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of people walking down a sidewalk. It does not contain any offensive content.

**Variant B:**
- `description`: The image captures a moment in time on a tree-lined street. A group of people are seen walking down the sidewalk, their shadows stretching out behind them. The photo is taken from a low angle, giving …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of people walking down a sidewalk. It does not contain any offensive content.
- `title`: A black and white photo of people walking down a sidewalk.

**Variant C:**
- `title`: A group of people walking down a sidewalk in a park.
- `description`: The image shows several individuals strolling along a path. The scene is set outdoors, with trees and buildings visible in the background. The photo appears to be black and white, adding a timeless qu…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000035.jpg

**Variant A:**
- `description`: This is a black and white photograph capturing a serene park scene. The image is taken from the perspective of someone standing on a sidewalk, looking towards a large building with many windows. A tre…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image does not contain any offensive content.

**Variant B:**
- `description`: This is a black and white photograph capturing a serene park scene. The image is taken from the perspective of someone standing on a sidewalk, looking towards a large building with many windows. A tre…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image does not contain any offensive content.
- `title`: A black and white photo of a park with trees, benches, and people.

**Variant C:**
- `title`: A black and white photograph of a park scene
- `description`: The image captures a serene park scene, with trees standing tall and people enjoying their time. The photograph is in black and white, adding a timeless quality to the scene.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000036.jpg

**Variant A:**
- `description`: The image is a black and white photograph capturing a moment in time. It features a group of people standing near a large truck, which is parked on a dirt road adjacent to a building. The building has…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of people standing near trees in front of a building. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph capturing a moment in time. It features a group of people standing near a large truck, which is parked on a dirt road adjacent to a building. The building has…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of people standing near trees in front of a building. It does not contain any offensive content.
- `title`: A black and white photo of a man standing next to a truck.

**Variant C:**
- `title`: A man in a truck is unloading something from the back of it.
- `description`: The image shows a man in a truck unloading something. The exact nature of what he's unloading cannot be determined from the image alone.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: medium

### 0000037.jpg

**Variant A:**
- `description`: This is a black and white photograph capturing a serene scene in a park. The image features a path that meanders through the park, inviting viewers to imagine walking down it. On either side of the pa…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of trees lining a path. It does not contain any offensive content.

**Variant B:**
- `description`: This is a black and white photograph capturing a serene scene in a park. The image features a path that meanders through the park, inviting viewers to imagine walking down it. On either side of the pa…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of trees lining a path. It does not contain any offensive content.
- `title`: A black and white photo of a tree lined street.

**Variant C:**
- `title`: A black and white photograph of a tree-lined street
- `description`: The image captures a serene scene of a tree-lined street, devoid of any human presence. The trees, standing tall on either side of the street, cast long shadows that stretch out into the distance. The…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000038.jpg

**Variant A:**
- `description`: The image is a black and white photograph capturing a serene park scene. Dominating the left side of the frame is a large tree, its branches spreading out in all directions, creating a canopy that add…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a large tree in a park. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph capturing a serene park scene. Dominating the left side of the frame is a large tree, its branches spreading out in all directions, creating a canopy that add…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a large tree in a park. It does not contain any offensive content.
- `title`: A tree with a street light in front of it.

**Variant C:**
- `title`: A large tree with a street lamp in front of it
- `description`: The image is black and white. The tree has many branches. There is a street lamp in front of the tree.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000039.jpg

**Variant A:**
- `description`: A black and white photo of a gas meter sign with people standing in front.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a stage performance from 1962. It does not contain any offensive content.

**Variant B:**
- `description`: A black and white photo of a gas meter sign with people standing in front.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a stage performance from 1962. It does not contain any offensive content.
- `title`: Gas meter

**Variant C:**
- `title`: 1926 Gas Meter
- `description`: A black and white photograph of a gas meter with people standing in front of it.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000040.jpg

**Variant A:**
- `description`: The image is a black and white photograph capturing a thrilling moment at a car race. The scene unfolds on a dirt track, where two cars are in the midst of a fierce competition. 

On the left side of …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph capturing a thrilling moment at a car race. The scene unfolds on a dirt track, where two cars are in the midst of a fierce competition. 

On the left side of …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A black and white photo of a car race

**Variant C:**
- `title`: So Cali Protest
- `description`: Protesters hold signs in front of a crowd. The image is black and white.
- `offensive`: False
- `offense_category`: Political
- `offense_reason`: —
- `confidence`: high

### 0000041.jpg

**Variant A:**
- `description`: The image is a black and white photograph of the [redacted institution] football stadium. The stadium has a large seating area in the center, with trees surrounding it. There are several cars parked a…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of the [redacted institution] stadium in Urbana, IL. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of the [redacted institution] football stadium. The stadium has a large seating area in the center, with trees surrounding it. There are several cars parked a…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of the [redacted institution] stadium in Urbana, IL. It does not contain any offensive content.
- `title`: A black and white photo of a stadium with cars in the parking lot.

**Variant C:**
- `title`: Aerial view of a stadium in Urmila, Illinois
- `description`: The image shows an aerial view of a large stadium located in Urmila, Illinois. The stadium is surrounded by several buildings and roads.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000042.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a farm. The farm consists of several buildings, including a large barn with a sloping roof and a smaller building with a flat roof. There are also trees sc…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of a farm. The farm consists of several buildings, including a large barn with a sloping roof and a smaller building with a flat roof. There are also trees sc…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image does not contain any offensive content.
- `title`: A black and white photo of a farm with several buildings.

**Variant C:**
- `title`: A black and white photograph of a farm with several buildings in the background.
- `description`: The photograph is in black and white and shows a farm with several buildings in the background. The image does not contain any offensive content.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000043.jpg

**Variant A:**
- `description`: The image is a black and white photograph of an old town. The buildings are made of brick, with some having flat roofs and others featuring sloping roofs. There's a large building in the center that s…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of an old town with buildings and trees. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of an old town. The buildings are made of brick, with some having flat roofs and others featuring sloping roofs. There's a large building in the center that s…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photograph of an old town with buildings and trees. It does not contain any offensive content.
- `title`: An old building with a dome on top.

**Variant C:**
- `title`: A black and white photograph of a town with various buildings including one labeled 'The Old Post Office'.
- `description`: The photograph is in black and white and shows several buildings, including one that is labeled 'The Old Post Office'. The image does not contain any offensive content.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000044.jpg

**Variant A:**
- `description`: The image is a black and white photograph of an old town. The buildings, which are the main focus of the image, are made of brick and have flat roofs. There are several trees scattered throughout the …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of an old town. The buildings, which are the main focus of the image, are made of brick and have flat roofs. There are several trees scattered throughout the …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A black and white photo of a town.

**Variant C:**
- `title`: A black and white photograph of a town with buildings and trees
- `description`: The image is a black and white photograph of a town. It features several buildings and trees in the background.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000045.jpg

**Variant A:**
- `description`: The image is a black and white photo of a man sitting on a bench in front of a building. The man is wearing a hat and holding a book. There are trees in the background, including one with a circular h…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photo of a man sitting on a bench in front of a building. The man is wearing a hat and holding a book. There are trees in the background, including one with a circular h…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A man sitting on a bench in front of a building.

**Variant C:**
- `title`: A man sitting on a bench in front of a building with trees and bushes around it.
- `description`: The image is black and white, showing a man sitting on a bench. The background consists of a building surrounded by trees and bushes.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000046.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a small town. The main focus is on the buildings, which are scattered throughout the scene. There are several trees in the background, adding to the rural …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a town with buildings and trees. It does not contain any offensive content.

**Variant B:**
- `description`: The image is a black and white photograph of a small town. The main focus is on the buildings, which are scattered throughout the scene. There are several trees in the background, adding to the rural …
- `offensive`: False
- `offense_category`: —
- `offense_reason`: The image is a black and white photo of a town with buildings and trees. It does not contain any offensive content.
- `title`: A black and white photo of a town with snow on the ground.

**Variant C:**
- `title`: A black and white photograph of a small town with a church in the background
- `description`: The photograph is taken from an aerial perspective, showing several buildings including a large building with a steeple. The town appears to be quiet and serene.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000047.jpg

**Variant A:**
- `description`: The image is a black and white photograph capturing a moment in time. It features a large crowd of people gathered in an open space, their attention directed towards the center where a stage is set up…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph capturing a moment in time. It features a large crowd of people gathered in an open space, their attention directed towards the center where a stage is set up…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A black and white photo of a crowd of people.

**Variant C:**
- `title`: A large crowd of people gathered in a park
- `description`: The image captures a moment where a group of people have come together to form a crowd. The setting is outdoors, with the backdrop featuring trees and a statue. The photo is taken from an elevated per…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000048.jpg

**Variant A:**
- `description`: The image is a black and white photograph of the entrance to Illinois Field. The gate, made of metal bars with decorative scrollwork on top, stands open. Above it, a sign reads "Illinois Field". To th…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of the entrance to Illinois Field. The gate, made of metal bars with decorative scrollwork on top, stands open. Above it, a sign reads "Illinois Field". To th…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: Illinois Field

**Variant C:**
- `title`: Illinois Field
- `description`: A black and white photo of a gate with the words Illinois Field above it.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000049.jpg

**Variant A:**
- `description`: A black and white photo of a protest with people holding signs that say "Stop the War".
- `offensive`: False
- `offense_category`: Other sensitive categories
- `offense_reason`: Protesters holding signs in the street.

**Variant B:**
- `description`: A black and white photo of a protest with people holding signs that say "Stop the War".
- `offensive`: False
- `offense_category`: Other sensitive categories
- `offense_reason`: Protesters holding signs in the street.
- `title`: Protesters hold signs that say "Stop the War".

**Variant C:**
- `title`: Protesters holding signs against the Vietnam War
- `description`: A group of people protesting the Vietnam War. They are holding up signs with messages such as 'Stop the War'. The photo is in black and white.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

### 0000050.jpg

**Variant A:**
- `description`: The image is a black and white photograph of a large brick building. The building has a tower on the left side, which houses a clock. The entrance to the building is located on the right side. There a…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —

**Variant B:**
- `description`: The image is a black and white photograph of a large brick building. The building has a tower on the left side, which houses a clock. The entrance to the building is located on the right side. There a…
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `title`: A large building with many windows.

**Variant C:**
- `title`: A large brick building with a porch and columns
- `description`: The building is made of bricks and has a porch with columns. It's a large structure, possibly an institution or government building.
- `offensive`: False
- `offense_category`: —
- `offense_reason`: —
- `confidence`: high

## Files

- `inference_timings.parquet` — one row per inference call (HTTP)
- `resource_samples.parquet` — GPU/CPU/memory traces
- `stage_timings.parquet` — one row per variant
- `field_parses.parquet` — one row per (image, variant, field)
- `outputs.jsonl` — raw model outputs for every call