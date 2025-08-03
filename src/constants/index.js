import {
  benefitIcon1,
  benefitIcon2,
  benefitIcon3,
  benefitIcon4,
  benefitImage2,
  chromecast,
  disc02,
  discord,
  discordBlack,
  facebook,
  figma,
  file02,
  framer,
  homeSmile,
  instagram,
  notification2,
  notification3,
  notification4,
  notion,
  photoshop,
  plusSquare,
  protopie,
  raindrop,
  recording01,
  recording03,
  roadmap1,
  roadmap2,
  roadmap3,
  roadmap4,
  searchMd,
  slack,
  sliders04,
  telegram,
  twitter,
  yourlogo,
} from "../assets";

export const navigation = [
  {
    id: "0",
    title: "Features",
    url: "#features",
  },
  {
    id: "1",
    title: "Pricing",
    url: "#pricing",
  },
  {
    id: "2",
    title: "How it works",
    url: "#how-to-use",
  },
  {
    id: "3",
    title: "Roadmap",
    url: "#roadmap",
  },
  {
    id: "4",
    title: "Upload Documents",
    url: "#signup",
    onlyMobile: true,
  },
  {
    id: "5",
    title: "Start Chatting",
    url: "#login",
    onlyMobile: true,
  },
];

export const heroIcons = [homeSmile, file02, searchMd, plusSquare];

export const notificationImages = [notification4, notification3, notification2];

export const companyLogos = [yourlogo, yourlogo, yourlogo, yourlogo, yourlogo];

export const neurofetchServices = [
  "Multi-format document support",
  "Instant semantic search",
  "Source citation & page numbers",
];

export const neurofetchServicesIcons = [
  recording03,
  recording01,
  disc02,
  chromecast,
  sliders04,
];

export const roadmap = [
  {
    id: "0",
    title: "Advanced Document Processing",
    text: "Enhanced support for complex document formats including scanned PDFs, tables, and structured data with improved OCR capabilities.",
    date: "May 2023",
    status: "done",
    imageUrl: roadmap1,
    colorful: true,
  },
  {
    id: "1",
    title: "Real-time Collaboration",
    text: "Enable multiple users to upload and query documents simultaneously with shared vector stores and collaborative annotations.",
    date: "May 2023",
    status: "progress",
    imageUrl: roadmap2,
  },
  {
    id: "2",
    title: "Custom Knowledge Bases",
    text: "Allow users to create and manage custom knowledge bases with domain-specific embeddings and specialized retrieval models.",
    date: "May 2023",
    status: "done",
    imageUrl: roadmap3,
  },
  {
    id: "3",
    title: "API Integration & Webhooks",
    text: "Provide comprehensive API access for integrating NeuroFetch into existing workflows with webhook support for real-time updates.",
    date: "May 2023",
    status: "progress",
    imageUrl: roadmap4,
  },
];

export const collabText =
  "With intelligent document processing and advanced semantic search, it's the perfect solution for teams looking to extract insights from their knowledge base.";

export const collabContent = [
  {
    id: "0",
    title: "Instant Document Indexing",
    text: collabText,
  },
  {
    id: "1",
    title: "Semantic Search Engine",
  },
  {
    id: "2",
    title: "Source Verification",
  },
];

export const collabApps = [
  {
    id: "0",
    title: "PDF Processing",
    icon: figma,
    width: 26,
    height: 36,
  },
  {
    id: "1",
    title: "DOCX Support",
    icon: notion,
    width: 34,
    height: 36,
  },
  {
    id: "2",
    title: "CSV Analysis",
    icon: discord,
    width: 36,
    height: 28,
  },
  {
    id: "3",
    title: "PPT Processing",
    icon: slack,
    width: 34,
    height: 35,
  },
  {
    id: "4",
    title: "Vector Store",
    icon: photoshop,
    width: 34,
    height: 34,
  },
  {
    id: "5",
    title: "Semantic Search",
    icon: protopie,
    width: 34,
    height: 34,
  },
  {
    id: "6",
    title: "API Access",
    icon: framer,
    width: 26,
    height: 34,
  },
  {
    id: "7",
    title: "Analytics",
    icon: raindrop,
    width: 38,
    height: 32,
  },
];

export const pricing = [
  {
    id: "0",
    title: "Starter",
    description: "Perfect for individual users and small document collections",
    price: "0",
    features: [
      "Upload up to 10 documents (PDF, DOCX, CSV, PPT)",
      "Basic semantic search with source citations",
      "Standard response time analytics",
      "Community support",
    ],
  },
  {
    id: "1",
    title: "Professional",
    description: "Advanced features for teams and larger document collections",
    price: "19.99",
    features: [
      "Upload up to 100 documents with unlimited storage",
      "Advanced RAG with custom embeddings",
      "Detailed analytics and usage insights",
      "Priority support and API access",
    ],
  },
  {
    id: "2",
    title: "Enterprise",
    description: "Custom solutions for large organizations with advanced security",
    price: null,
    features: [
      "Unlimited document uploads and storage",
      "Custom knowledge base deployment",
      "Advanced security and compliance features",
      "Dedicated support and custom integrations",
    ],
  },
];

export const benefits = [
  {
    id: "0",
    title: "Multi-Format Support",
    text: "Upload and process PDFs, DOCX files, CSV data, and PowerPoint presentations with instant indexing and embedding generation.",
    backgroundUrl: "./src/assets/benefits/card-1.svg",
    iconUrl: benefitIcon1,
    imageUrl: benefitImage2,
  },
  {
    id: "1",
    title: "Semantic Search",
    text: "Advanced RAG technology with vector embeddings enables precise semantic search across your entire document knowledge base.",
    backgroundUrl: "./src/assets/benefits/card-2.svg",
    iconUrl: benefitIcon2,
    imageUrl: benefitImage2,
    light: true,
  },
  {
    id: "2",
    title: "Source Citation",
    text: "Get precise answers with exact source file references, page numbers, and response time analytics for complete transparency.",
    backgroundUrl: "./src/assets/benefits/card-3.svg",
    iconUrl: benefitIcon3,
    imageUrl: benefitImage2,
  },
  {
    id: "3",
    title: "Instant Processing",
    text: "Documents are processed and indexed immediately upon upload, with embeddings added to the vector store for instant querying.",
    backgroundUrl: "./src/assets/benefits/card-4.svg",
    iconUrl: benefitIcon4,
    imageUrl: benefitImage2,
    light: true,
  },
  {
    id: "4",
    title: "Advanced Analytics",
    text: "Track query performance, response times, and document usage patterns with comprehensive analytics and insights.",
    backgroundUrl: "./src/assets/benefits/card-5.svg",
    iconUrl: benefitIcon1,
    imageUrl: benefitImage2,
  },
  {
    id: "5",
    title: "Secure & Private",
    text: "Enterprise-grade security with encrypted document storage and private vector databases for your sensitive information.",
    backgroundUrl: "./src/assets/benefits/card-6.svg",
    iconUrl: benefitIcon2,
    imageUrl: benefitImage2,
  },
];

export const socials = [
  {
    id: "0",
    title: "Discord",
    iconUrl: discordBlack,
    url: "#",
  },
  {
    id: "1",
    title: "Twitter",
    iconUrl: twitter,
    url: "#",
  },
  {
    id: "2",
    title: "Instagram",
    iconUrl: instagram,
    url: "#",
  },
  {
    id: "3",
    title: "Telegram",
    iconUrl: telegram,
    url: "#",
  },
  {
    id: "4",
    title: "Facebook",
    iconUrl: facebook,
    url: "#",
  },
];
