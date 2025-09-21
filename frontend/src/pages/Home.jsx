import React from 'react'
import { Link } from 'react-router-dom'
import { Leaf, Shield, BarChart3, Upload } from 'lucide-react'
import { motion } from 'framer-motion'

const Home = () => {
  const features = [
    {
      icon: <Upload className="h-8 w-8" />,
      title: "Easy Image Upload",
      description: "Simply upload an image of a cotton leaf and get instant analysis."
    },
    {
      icon: <Shield className="h-8 w-8" />,
      title: "Accurate Detection",
      description: "Our AI model accurately identifies various cotton leaf diseases."
    },
    {
      icon: <BarChart3 className="h-8 w-8" />,
      title: "Detailed Analysis",
      description: "Get comprehensive reports and treatment recommendations."
    }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-green-50 to-blue-50 opacity-90"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="flex justify-center mb-6"
            >
              <div className="inline-flex items-center justify-center p-2 bg-green-100 rounded-full">
                <Leaf className="h-8 w-8 text-green-600" />
              </div>
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            >
              Protect Your Cotton Crops with AI
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="text-xl text-gray-600 mb-10 max-w-3xl mx-auto"
            >
              Detect diseases in cotton leaves early with our advanced deep learning technology. 
              Get accurate diagnoses and treatment recommendations to save your crops.
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="flex flex-col sm:flex-row gap-4 justify-center"
            >
              <Link
                to="/detect"
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-green-600 hover:bg-green-700 transition-colors"
              >
                <Upload className="mr-2 h-5 w-5" />
                Detect Disease Now
              </Link>
              <Link
                to="/about"
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-green-700 bg-green-100 hover:bg-green-200 transition-colors"
              >
                Learn More
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our AI-powered system makes cotton leaf disease detection simple and accurate
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                className="text-center p-6 bg-gray-50 rounded-lg hover:shadow-md transition-shadow"
              >
                <div className="inline-flex items-center justify-center p-3 bg-green-100 rounded-full text-green-600 mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-r from-green-50 to-blue-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="p-6"
            >
              <div className="text-4xl font-bold text-green-600 mb-2">95%</div>
              <div className="text-gray-600">Accuracy Rate</div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="p-6"
            >
              <div className="text-4xl font-bold text-green-600 mb-2">5+</div>
              <div className="text-gray-600">Diseases Detected</div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="p-6"
            >
              <div className="text-4xl font-bold text-green-600 mb-2">24/7</div>
              <div className="text-gray-600">Availability</div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Ready to Protect Your Crops?</h2>
          <p className="text-lg text-gray-600 mb-8">
            Join hundreds of farmers who are already using our technology to save their cotton crops
          </p>
          <Link
            to="/detect"
            className="inline-flex items-center px-8 py-4 border border-transparent text-lg font-medium rounded-md text-white bg-green-600 hover:bg-green-700 transition-colors"
          >
            <Upload className="mr-2 h-6 w-6" />
            Start Detection Now
          </Link>
        </div>
      </section>
    </div>
  )
}

export default Home