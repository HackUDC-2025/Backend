# ArtLens - Interactive Museum Guide Powered by AI

**Project developed for HackUDC 2025**

ArtLens is a mobile application designed to transform the museum experience by offering an interactive and personalized guide based on the user's context. Using advanced Artificial Intelligence techniques, such as artwork recognition and enriched description generation, ArtLens focuses on the Museo del Prado, though its architecture allows adaptation to other museums.

## What is ArtLens?

ArtLens is a mobile application that acts as an interactive and personalized museum guide. Its name reflects its purpose: to be a "lens" that enhances and enriches the user's museum experience using artificial intelligence technology.

## What does ArtLens do? What problem does it solve?

ArtLens solves the problem of visiting a museum without a guide or audio guide by offering an enriched and contextually adapted experience. Often, visitors lack access to specialized guides or cannot fully appreciate their visit due to a lack of contextualized information. ArtLens addresses this by providing:

- **Detailed and personalized descriptions**: Generated in real-time using AI, tailored to the user's context.
- **Automatic audio guides**: Converts descriptions into audio using **gTTS**, ideal for those who prefer listening over reading.
- **Personalized recommendations**: Based on the artworks you are most interested in, thanks to embedding retrieval with **Milvus**.

ArtLens not only informs you about artworks but does so in a way that adapts to your interests and needs, enriching your museum experience.

## What does ArtLens depend on? What do I need to install and use it?

To install and run ArtLens, follow these steps:

1. **Clone the repositories**:
   - Frontend: [https://github.com/HackUDC-2025/Frontend](https://github.com/HackUDC-2025/Frontend)
   - Backend: [https://github.com/HackUDC-2025/Backend](https://github.com/HackUDC-2025/Backend)

2. **Set up services with Docker**:
   - Ensure Docker and Docker Compose are installed.
   - Run the following commands to start the **Ollama** and **Milvus** services:
     ```bash
     docker-compose up -d
     ```

3. **Configure and run the frontend and backend**:
   - Follow the detailed instructions in the respective repositories to set up and run both the frontend and backend.

## How does ArtLens work?

ArtLens uses a combination of AI technologies to deliver a unique experience:

1. **Artwork recognition**: We use the **CLIP** model to identify artworks in real-time through the device's camera.
2. **Description generation**: We employ the open-source **LLama 3.2** model (with a modified Apache 2.0 license) to generate contextualized and enriched descriptions of artworks.
3. **Embedding retrieval**: With **Milvus**, we perform efficient embedding searches to find similar artworks based on your interests.
4. **Text-to-Speech**: We integrate **gTTS** to convert descriptions into audio, enabling personalized audio guides.

## What works and what doesn’t?

### What works:
- All core functionalities are implemented and working:
  - Artwork recognition.
  - Contextualized description generation.
  - Text-to-audio conversion for audio guides.
  - Historical search maintenance.

### What doesn’t work (or has limitations):
- **Limited artwork coverage**: Currently, ArtLens does not include all artworks from the Museo del Prado. This is due to the lack of an official dataset, so we have used web scraping techniques to gather information on a small portion of the artworks. We are working to expand this coverage in future versions.

## Where can I ask if I find an issue or have questions?

If you encounter an issue or have questions, you can:
- Open an **issue** in the corresponding repository (Frontend or Backend).
- Email the project creators:
  - Alejandro Dopico: alejandro.dopico2@udc.es
  - Mario Diez: mario.diez@udc.es
  - Antón Canzobre: antoncanzobremar@gmail.com

## How can I contribute?

We would love for you to contribute to ArtLens! To do so, follow these steps:
1. Fork the repository.
2. Create a new branch with your feature name (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a **Pull Request** for us to review your changes.

You can also contact the project creators directly to discuss ideas or collaborations.

## Who has contributed?

ArtLens was developed by the three team members during the HackUDC 2025 event:
- Alejandro Dopico
- Mario Diez
- Antón Canzobre

## What license does ArtLens have?

ArtLens is licensed under the **MIT License**. We chose the MIT License because it is simple, permissive, and widely accepted in the open-source community. It allows others to use, modify, and distribute our code without restrictions, fostering adoption and collaboration. Additionally, it is compatible with the licenses of our dependencies, such as Milvus (Apache 2.0) and Angular (MIT). While we considered Apache 2.0 and GPL, the MIT License aligns better with our goals of maximizing adoption and maintaining project simplicity.

## Acknowledgments

We thank all contributors, sponsors, and the HackUDC 2025 organizers for making this project possible.
