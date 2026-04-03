// ============================================
// SETUP SCRIPT
//
// Generates sample documents in 5 formats so
// you can test every phase immediately.
//
// Run: npm run setup
//
// Creates:
//   sample-docs/world-countries.txt
//   sample-docs/scientists.md
//   sample-docs/cities-data.csv
//   sample-docs/inventions.json
//   sample-docs/solar-system.txt
//   data/  (empty directory for vector stores)
// ============================================

import * as fs from "fs";
import * as path from "path";

const SAMPLE_DIR = "sample-docs";
const DATA_DIR = "data";

// ============================================
// SAMPLE 1: World Countries (.txt)
// ============================================

const worldCountries = `World Countries — Facts and Capitals

Canada
Capital: Ottawa
Population: 40 million
Official Languages: English, French
Fun Fact: Canada has the longest coastline in the world at 202,080 km. It is the second-largest country by total area.

United States
Capital: Washington, D.C.
Population: 331 million
Official Language: English (de facto)
Fun Fact: The US has 50 states and is home to the Grand Canyon, one of the seven natural wonders of the world.

Brazil
Capital: Brasilia
Population: 215 million
Official Language: Portuguese
Fun Fact: Brazil contains about 60% of the Amazon rainforest, the largest tropical rainforest on Earth.

India
Capital: New Delhi
Population: 1.4 billion
Official Languages: Hindi, English
Fun Fact: India is the world's largest democracy and home to the Taj Mahal, one of the seven wonders of the modern world.

Japan
Capital: Tokyo
Population: 125 million
Official Language: Japanese
Fun Fact: Japan consists of 6,852 islands and has the world's oldest company, Kongo Gumi, founded in 578 AD.

Germany
Capital: Berlin
Population: 84 million
Official Language: German
Fun Fact: Germany is known as the "Land of Poets and Thinkers" and is home to over 1,500 different beers.

Australia
Capital: Canberra
Population: 26 million
Official Language: English
Fun Fact: Australia is both a country and a continent. The Great Barrier Reef is the largest living structure on Earth.

France
Capital: Paris
Population: 68 million
Official Language: French
Fun Fact: France is the most visited country in the world with about 90 million tourists per year. The Eiffel Tower was supposed to be temporary.

Egypt
Capital: Cairo
Population: 104 million
Official Language: Arabic
Fun Fact: The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient World and the only one still standing.

South Korea
Capital: Seoul
Population: 52 million
Official Language: Korean
Fun Fact: South Korea has the fastest internet speed in the world. K-pop and Korean dramas have become global cultural phenomena.

Mexico
Capital: Mexico City
Population: 129 million
Official Language: Spanish
Fun Fact: Mexico City was built on the ruins of Tenochtitlan, the ancient Aztec capital. Mexico introduced chocolate to the world.

Kenya
Capital: Nairobi
Population: 54 million
Official Languages: Swahili, English
Fun Fact: Kenya is famous for its wildlife safaris and is home to the Great Wildebeest Migration in the Masai Mara.

Norway
Capital: Oslo
Population: 5.5 million
Official Language: Norwegian
Fun Fact: Norway has the longest road tunnel in the world (24.5 km) and is famous for the Northern Lights (Aurora Borealis).

Argentina
Capital: Buenos Aires
Population: 46 million
Official Language: Spanish
Fun Fact: Argentina is the birthplace of tango and has won the FIFA World Cup three times (1978, 1986, 2022).

Thailand
Capital: Bangkok
Population: 72 million
Official Language: Thai
Fun Fact: Thailand is the only Southeast Asian country that was never colonized by a European power. Its name means "Land of the Free."

Nigeria
Capital: Abuja
Population: 223 million
Official Language: English
Fun Fact: Nigeria is Africa's most populous country and has over 500 distinct languages spoken within its borders.

New Zealand
Capital: Wellington
Population: 5.1 million
Official Languages: English, Maori
Fun Fact: New Zealand was the first country to give women the right to vote in 1893. It has more sheep than people.

Turkey
Capital: Ankara
Population: 85 million
Official Language: Turkish
Fun Fact: Istanbul is the only city in the world that spans two continents — Europe and Asia. Turkey introduced coffee to Europe.

Peru
Capital: Lima
Population: 34 million
Official Languages: Spanish, Quechua
Fun Fact: Peru is home to Machu Picchu, the 15th-century Inca citadel set high in the Andes Mountains.

Sweden
Capital: Stockholm
Population: 10.5 million
Official Language: Swedish
Fun Fact: Sweden has an Ice Hotel that is rebuilt every winter. The Nobel Prize was established by Swedish inventor Alfred Nobel.
`;

// ============================================
// SAMPLE 2: Famous Scientists (.md)
// ============================================

const scientists = `# Famous Scientists Who Changed the World

## Physics

### Albert Einstein (1879–1955)
- **Nationality**: German-born, later American
- **Key Discovery**: Theory of Relativity (E=mc²)
- **Nobel Prize**: 1921 (Photoelectric Effect)
- **Impact**: Revolutionized our understanding of space, time, gravity, and the universe. His work laid the foundation for nuclear energy and GPS technology.

### Isaac Newton (1643–1727)
- **Nationality**: English
- **Key Discovery**: Laws of Motion and Universal Gravitation
- **Notable Work**: *Principia Mathematica* (1687)
- **Impact**: Founded classical mechanics. Invented calculus (independently of Leibniz). Discovered that white light is composed of a spectrum of colors.

### Marie Curie (1867–1934)
- **Nationality**: Polish-born, later French
- **Key Discovery**: Radioactivity, Polonium, and Radium
- **Nobel Prizes**: 1903 (Physics) and 1911 (Chemistry)
- **Impact**: First woman to win a Nobel Prize. First person to win Nobel Prizes in two different sciences. Her research led to advances in cancer treatment.

### Nikola Tesla (1856–1943)
- **Nationality**: Serbian-American
- **Key Discovery**: Alternating Current (AC) electrical system
- **Notable Work**: Tesla coil, rotating magnetic field
- **Impact**: His AC system powers the modern world. Contributed to radio technology, X-ray imaging, and remote control systems.

## Biology & Medicine

### Charles Darwin (1809–1882)
- **Nationality**: English
- **Key Discovery**: Theory of Evolution by Natural Selection
- **Notable Work**: *On the Origin of Species* (1859)
- **Impact**: Fundamentally changed our understanding of life on Earth. Established that all species descend from common ancestors.

### Louis Pasteur (1822–1895)
- **Nationality**: French
- **Key Discovery**: Germ Theory of Disease, Pasteurization
- **Notable Work**: Developed vaccines for rabies and anthrax
- **Impact**: Saved countless lives through pasteurization and vaccination. Founded the field of microbiology.

### Rosalind Franklin (1920–1958)
- **Nationality**: English
- **Key Discovery**: X-ray diffraction images of DNA (Photo 51)
- **Notable Work**: Critical contribution to understanding DNA structure
- **Impact**: Her work was essential to Watson and Crick's discovery of the DNA double helix. She also made important contributions to understanding viruses.

## Chemistry

### Dmitri Mendeleev (1834–1907)
- **Nationality**: Russian
- **Key Discovery**: Periodic Table of Elements
- **Notable Work**: Predicted properties of undiscovered elements
- **Impact**: Organized all known elements into a systematic table that predicted the existence of new elements before they were discovered.

### Linus Pauling (1901–1994)
- **Nationality**: American
- **Key Discovery**: Nature of the Chemical Bond
- **Nobel Prizes**: 1954 (Chemistry) and 1962 (Peace)
- **Impact**: Only person to win two unshared Nobel Prizes. His work on chemical bonds is foundational to modern chemistry and molecular biology.

## Computer Science & Mathematics

### Alan Turing (1912–1954)
- **Nationality**: English
- **Key Discovery**: Turing Machine (theoretical computer)
- **Notable Work**: Breaking the Enigma code in WWII
- **Impact**: Father of theoretical computer science and artificial intelligence. His concepts underpin every modern computer.

### Ada Lovelace (1815–1852)
- **Nationality**: English
- **Key Discovery**: First computer algorithm
- **Notable Work**: Notes on Charles Babbage's Analytical Engine
- **Impact**: Recognized as the first computer programmer. She envisioned that computers could go beyond pure calculation to create music and art.

### Grace Hopper (1906–1992)
- **Nationality**: American
- **Key Discovery**: First compiler for a computer programming language
- **Notable Work**: Developed COBOL programming language
- **Impact**: Pioneered the concept that programming languages could be written in English-like syntax. Popularized the term "debugging" after finding an actual moth in a computer.
`;

// ============================================
// SAMPLE 3: Cities Data (.csv)
// ============================================

const citiesData = `city,country,population,continent,founded,area_km2,famous_for
Tokyo,Japan,13960000,Asia,1457,2194,Technology and cherry blossoms
London,United Kingdom,8982000,Europe,43,1572,Big Ben and the British Museum
New York,United States,8336000,North America,1624,783,Statue of Liberty and Wall Street
Paris,France,2161000,Europe,250,105,Eiffel Tower and the Louvre
Sydney,Australia,5312000,Oceania,1788,12368,Opera House and Harbour Bridge
Cairo,Egypt,10100000,Africa,-3100,528,Pyramids of Giza and the Sphinx
Mumbai,India,12478000,Asia,1507,603,Bollywood and Gateway of India
São Paulo,Brazil,12330000,South America,1554,1521,Cultural diversity and street art
Istanbul,Turkey,15460000,Europe/Asia,-660,5343,Hagia Sophia and the Grand Bazaar
Toronto,Canada,2794000,North America,1793,630,CN Tower and multiculturalism
Dubai,United Arab Emirates,3490000,Asia,1833,4114,Burj Khalifa and luxury shopping
Singapore,Singapore,5454000,Asia,1819,733,Gardens by the Bay and street food
Rome,Italy,2873000,Europe,-753,1285,Colosseum and Vatican City
Nairobi,Kenya,4397000,Africa,1899,696,Wildlife safaris and national parks
Stockholm,Sweden,975000,Europe,1252,188,Nobel Prize and archipelago islands
`;

// ============================================
// SAMPLE 4: Inventions (.json)
// ============================================

const inventions = [
  {
    name: "Printing Press",
    year: 1440,
    inventor: "Johannes Gutenberg",
    country: "Germany",
    description: "A mechanical device for transferring ink from movable type to paper, enabling mass production of books.",
    impact: "Democratized knowledge, accelerated the Renaissance, and made literacy accessible to the masses. Before the printing press, books were copied by hand and only the wealthy could afford them."
  },
  {
    name: "Steam Engine",
    year: 1712,
    inventor: "Thomas Newcomen (improved by James Watt)",
    country: "England",
    description: "A heat engine that converts steam pressure into mechanical work.",
    impact: "Powered the Industrial Revolution. Enabled factories, railways, and steamships. Transformed society from agrarian to industrial."
  },
  {
    name: "Telephone",
    year: 1876,
    inventor: "Alexander Graham Bell",
    country: "United States",
    description: "A device that converts sound into electrical signals for transmission over distances.",
    impact: "Revolutionized human communication. Led to the telecommunications industry and eventually to the internet and smartphones."
  },
  {
    name: "Light Bulb",
    year: 1879,
    inventor: "Thomas Edison",
    country: "United States",
    description: "A practical incandescent light source powered by electricity.",
    impact: "Extended productive hours beyond daylight. Transformed cities, workplaces, and homes. Led to the modern electrical grid."
  },
  {
    name: "Airplane",
    year: 1903,
    inventor: "Wright Brothers (Orville and Wilbur)",
    country: "United States",
    description: "The first successful powered, heavier-than-air flying machine.",
    impact: "Enabled rapid global travel and trade. Shrunk the world by making distant places reachable in hours instead of weeks."
  },
  {
    name: "Penicillin",
    year: 1928,
    inventor: "Alexander Fleming",
    country: "Scotland",
    description: "The first true antibiotic, derived from the Penicillium mold.",
    impact: "Saved hundreds of millions of lives by treating bacterial infections. Launched the era of antibiotics and modern medicine."
  },
  {
    name: "Transistor",
    year: 1947,
    inventor: "John Bardeen, Walter Brattain, William Shockley",
    country: "United States",
    description: "A semiconductor device that amplifies or switches electronic signals.",
    impact: "Foundation of all modern electronics. Enabled computers, smartphones, and the digital age. Replaced vacuum tubes."
  },
  {
    name: "Internet",
    year: 1969,
    inventor: "Vint Cerf, Bob Kahn (TCP/IP protocol)",
    country: "United States",
    description: "A global network of interconnected computers that communicate using standardized protocols.",
    impact: "Transformed every aspect of modern life — communication, commerce, education, entertainment. Connected billions of people worldwide."
  },
  {
    name: "World Wide Web",
    year: 1989,
    inventor: "Tim Berners-Lee",
    country: "United Kingdom (at CERN, Switzerland)",
    description: "A system of interlinked hypertext documents accessed via the internet using a web browser.",
    impact: "Made the internet accessible to everyone. Created the foundation for e-commerce, social media, and the modern digital economy."
  },
  {
    name: "CRISPR Gene Editing",
    year: 2012,
    inventor: "Jennifer Doudna, Emmanuelle Charpentier",
    country: "United States / France",
    description: "A molecular tool that allows precise editing of DNA sequences in living organisms.",
    impact: "Revolutionized genetics and medicine. Potential to cure genetic diseases, improve crops, and transform biotechnology. Won the 2020 Nobel Prize in Chemistry."
  }
];

// ============================================
// SAMPLE 5: Solar System (.txt)
// ============================================

const solarSystem = `The Solar System — A Guide to Our Cosmic Neighborhood

The Sun
Type: G-type main-sequence star (Yellow Dwarf)
Age: 4.6 billion years
Diameter: 1,391,000 km
Temperature: 5,500°C (surface), 15 million°C (core)
The Sun contains 99.86% of the total mass of the Solar System. It converts 600 million tons of hydrogen into helium every second through nuclear fusion.

Mercury
Position: 1st planet from the Sun
Type: Terrestrial (rocky)
Diameter: 4,879 km
Distance from Sun: 57.9 million km
Day Length: 59 Earth days
Year Length: 88 Earth days
Moons: 0
Mercury is the smallest planet and has the most extreme temperature swings — from -180°C at night to 430°C during the day. Despite being closest to the Sun, it is not the hottest planet.

Venus
Position: 2nd planet from the Sun
Type: Terrestrial (rocky)
Diameter: 12,104 km
Distance from Sun: 108.2 million km
Day Length: 243 Earth days (rotates backward!)
Year Length: 225 Earth days
Moons: 0
Venus is the hottest planet (465°C) due to its thick CO2 atmosphere creating a runaway greenhouse effect. A day on Venus is longer than its year, and it rotates in the opposite direction to most planets.

Earth
Position: 3rd planet from the Sun
Type: Terrestrial (rocky)
Diameter: 12,742 km
Distance from Sun: 149.6 million km (1 AU)
Day Length: 24 hours
Year Length: 365.25 days
Moons: 1 (The Moon)
Earth is the only known planet to support life. It has liquid water on its surface, a protective magnetic field, and an atmosphere rich in nitrogen and oxygen. The Moon stabilizes Earth's axial tilt, creating predictable seasons.

Mars
Position: 4th planet from the Sun
Type: Terrestrial (rocky)
Diameter: 6,779 km
Distance from Sun: 227.9 million km
Day Length: 24 hours 37 minutes
Year Length: 687 Earth days
Moons: 2 (Phobos and Deimos)
Mars is called the Red Planet due to iron oxide (rust) on its surface. It has the tallest volcano (Olympus Mons, 21.9 km) and the deepest canyon (Valles Marineris, 7 km deep) in the Solar System. NASA's rovers have found evidence of ancient water.

Jupiter
Position: 5th planet from the Sun
Type: Gas Giant
Diameter: 139,820 km
Distance from Sun: 778.5 million km
Day Length: 10 hours (fastest rotation)
Year Length: 12 Earth years
Moons: 95 known (including Ganymede, the largest moon in the Solar System)
Jupiter is the largest planet — you could fit 1,300 Earths inside it. The Great Red Spot is a storm larger than Earth that has been raging for at least 350 years. Jupiter's strong gravity protects inner planets by capturing many asteroids and comets.

Saturn
Position: 6th planet from the Sun
Type: Gas Giant
Diameter: 116,460 km
Distance from Sun: 1.43 billion km
Day Length: 10.7 hours
Year Length: 29 Earth years
Moons: 146 known (including Titan, which has a thick atmosphere and methane lakes)
Saturn is famous for its spectacular ring system, made of billions of ice and rock particles. Saturn is so light it would float in water (density 0.687 g/cm³). Its moon Titan is the only moon with a substantial atmosphere.

Uranus
Position: 7th planet from the Sun
Type: Ice Giant
Diameter: 50,724 km
Distance from Sun: 2.87 billion km
Day Length: 17.2 hours
Year Length: 84 Earth years
Moons: 27 known
Uranus rotates on its side (98° tilt), likely due to a massive ancient collision. It has faint rings discovered in 1977. Its blue-green color comes from methane in its atmosphere. Uranus is the coldest planet with temperatures reaching -224°C.

Neptune
Position: 8th planet from the Sun
Type: Ice Giant
Diameter: 49,528 km
Distance from Sun: 4.5 billion km
Day Length: 16.1 hours
Year Length: 165 Earth years
Moons: 16 known (including Triton, which orbits backward)
Neptune has the strongest winds in the Solar System, reaching 2,100 km/h. It was the first planet predicted mathematically before being observed. Its moon Triton is one of the coldest objects in the Solar System (-235°C).

Dwarf Planets
The Solar System also contains five recognized dwarf planets: Pluto, Eris, Haumea, Makemake, and Ceres. Pluto was reclassified from a planet to a dwarf planet in 2006 by the International Astronomical Union. Ceres is located in the asteroid belt between Mars and Jupiter and is the only dwarf planet in the inner Solar System.
`;

// ============================================
// GENERATE ALL FILES
// ============================================

function setup() {
  console.log("\n--- Setup: Generating sample documents ---\n");

  // Create directories
  if (!fs.existsSync(SAMPLE_DIR)) fs.mkdirSync(SAMPLE_DIR, { recursive: true });
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

  // Write files
  const files = [
    { name: "world-countries.txt", content: worldCountries, format: ".txt" },
    { name: "scientists.md", content: scientists, format: ".md" },
    { name: "cities-data.csv", content: citiesData, format: ".csv" },
    { name: "inventions.json", content: JSON.stringify(inventions, null, 2), format: ".json" },
    { name: "solar-system.txt", content: solarSystem, format: ".txt" },
  ];

  for (const file of files) {
    const filePath = path.join(SAMPLE_DIR, file.name);
    fs.writeFileSync(filePath, file.content, "utf-8");
    const sizeKB = (Buffer.byteLength(file.content) / 1024).toFixed(1);
    console.log(`  Created ${filePath} (${sizeKB} KB) [${file.format}]`);
  }

  // Write sample-docs README
  const sampleReadme = `# Sample Documents

These files were auto-generated by \`npm run setup\` for testing the RAG pipeline.

| File | Format | Content | Good for testing |
|------|--------|---------|-----------------|
| world-countries.txt | .txt | 20 countries with facts | Factual Q&A, keyword search |
| scientists.md | .md | 12 famous scientists | Markdown parsing, structured content |
| cities-data.csv | .csv | 15 cities with data | CSV parsing, tabular data |
| inventions.json | .json | 10 major inventions | JSON parsing, nested data |
| solar-system.txt | .txt | 9 planets + Sun | Multi-file RAG, numerical data |

## Adding your own files

Drop any .txt, .md, .csv, .json, .pdf, or .docx file into this folder.
Then run Phase 5 to ingest them into the vector store.

## Testing PDF support

PDF generation requires extra tools, so no sample PDF is included.
To test PDF support, add any .pdf file to this folder and ingest it in Phase 5.
`;

  fs.writeFileSync(path.join(SAMPLE_DIR, "README.md"), sampleReadme, "utf-8");
  console.log(`  Created ${SAMPLE_DIR}/README.md`);

  console.log(`\n  Done! ${files.length} sample documents created in ${SAMPLE_DIR}/`);
  console.log(`  Empty data/ directory ready for vector stores.\n`);
  console.log("  Next steps:");
  console.log("    Phases 1-4 (local):  npm run phase1");
  console.log("    Phases 5-8 (cloud):  cp .env.example .env  (add your OpenRouter key)");
  console.log("                         npm run phase5\n");
}

setup();
