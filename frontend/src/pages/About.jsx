import { Leaf, Users, Target, Heart, Code, Database } from 'lucide-react'
import { motion } from 'framer-motion'
import React from 'react'

const About = () => {
  const team = [
    {
      name: "AI Research Team",
      role: "Deep Learning Experts",
      icon: <Code className="h-8 w-8" />
    },
    {
      name: "Agricultural Specialists",
      role: "Plant Pathologists",
      icon: <Leaf className="h-8 w-8" />
    },
    {
      name: "Data Engineers",
      role: "Dataset Management",
      icon: <Database className="h-8 w-8" />
    }
  ]

  const diseases = [
    {
      name: "Bacterial Blight",
      description: "Caused by Xanthomonas bacteria, causing angular leaf spots"
    },
    {
      name: "Curly Virus",
      description: "Viral infection causing leaf curling and stunted growth"
    },
    {
      name: "Fusarium Wilt",
      description: "Fungal disease causing wilting and vascular discoloration"
    },
    {
      name: "Powdery Mildew",
      description: "Fungal infection creating white powdery spots on leaves"
    },
    {
      name: "Target Spot",
      description: "Fungal disease causing concentric ring spots on leaves"
    },
    {
      name: "Healthy Leaves",
      description: "Properly growing cotton plants without disease symptoms"
    }
  ]

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold text-gray-800 mb-4">About CottonGuard</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Revolutionizing cotton farming through AI-powered disease detection and prevention
        </p>
      </motion.div>

      {/* Mission Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="mb-16"
      >
        <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-2xl p-8">
          <div className="flex items-center mb-6">
            <Target className="h-8 w-8 text-green-600 mr-3" />
            <h2 className="text-2xl font-bold text-gray-800">Our Mission</h2>
          </div>
          <p className="text-gray-700 text-lg leading-relaxed">
            At CottonGuard, we're committed to helping farmers protect their cotton crops through 
            early disease detection. Our AI-powered platform uses advanced deep learning algorithms 
            to identify diseases quickly and accurately, enabling timely intervention and treatment.
          </p>
        </div>
      </motion.section>

      {/* Technology Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4 }}
        className="mb-16"
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">How It Works</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center p-6 bg-white rounded-lg shadow-md">
            <div className="inline-flex items-center justify-center p-3 bg-blue-100 rounded-full text-blue-600 mb-4">
              <Database className="h-8 w-8" />
            </div>
            <h3 className="text-xl font-semibold mb-2">1. Data Collection</h3>
            <p className="text-gray-600">
              We've compiled thousands of labeled cotton leaf images to train our AI models
            </p>
          </div>

          <div className="text-center p-6 bg-white rounded-lg shadow-md">
            <div className="inline-flex items-center justify-center p-3 bg-green-100 rounded-full text-green-600 mb-4">
              <Code className="h-8 w-8" />
            </div>
            <h3 className="text-xl font-semibold mb-2">2. AI Training</h3>
            <p className="text-gray-600">
              Our deep learning models are trained to recognize patterns and disease signatures
            </p>
          </div>

          <div className="text-center p-6 bg-white rounded-lg shadow-md">
            <div className="inline-flex items-center justify-center p-3 bg-purple-100 rounded-full text-purple-600 mb-4">
              <Leaf className="h-8 w-8" />
            </div>
            <h3 className="text-xl font-semibold mb-2">3. Detection</h3>
            <p className="text-gray-600">
              Farmers upload images and receive instant, accurate disease analysis
            </p>
          </div>
        </div>
      </motion.section>

      {/* Diseases Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.6 }}
        className="mb-16"
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">Diseases We Detect</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {diseases.map((disease, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">{disease.name}</h3>
              <p className="text-gray-600 text-sm">{disease.description}</p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Team Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.8 }}
        className="mb-16"
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">Our Team</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          {team.map((member, index) => (
            <div key={index} className="text-center p-6 bg-white rounded-lg shadow-md">
              <div className="inline-flex items-center justify-center p-3 bg-gray-100 rounded-full text-gray-600 mb-4">
                {member.icon}
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-1">{member.name}</h3>
              <p className="text-gray-600">{member.role}</p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Impact Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1 }}
        className="bg-green-50 rounded-2xl p-8"
      >
        <div className="flex items-center mb-6">
          <Heart className="h-8 w-8 text-green-600 mr-3" />
          <h2 className="text-2xl font-bold text-gray-800">Our Impact</h2>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-3">For Farmers</h3>
            <ul className="space-y-2 text-gray-700">
              <li>• Early disease detection saves crops</li>
              <li>• Reduced pesticide use through targeted treatment</li>
              <li>• Increased yield and profitability</li>
              <li>• 24/7 access to expert-level analysis</li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-3">For Agriculture</h3>
            <ul className="space-y-2 text-gray-700">
              <li>• Sustainable farming practices</li>
              <li>• Reduced crop loss and waste</li>
              <li>• Data-driven insights for researchers</li>
              <li>• Global food security improvement</li>
            </ul>
          </div>
        </div>
      </motion.section>
    </div>
  )
}

export default About