import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Section from "../components/Section";
import Button from "../components/Button";
import Gradient from "../components/design/Gradient";

const Signup = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState({ type: "", text: "" });
  const navigate = useNavigate();

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => {
      setMessage({ type: "", text: "" });
    }, 5000);
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    if (password.length < 8) {
      showMessage("error", "Password must be at least 8 characters long.");
      return;
    }
    if (password !== confirmPassword) {
      showMessage("error", "Passwords do not match");
      return;
    }

    try {
      const response = await fetch(`${import.meta.env.VITE_AUTH_API_URL || "http://localhost:3000"}/api/auth/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        showMessage("success", "Signup successful! Please login.");
        setTimeout(() => {
          navigate("/login");
        }, 1000);
      } else {
        showMessage("error", data.message || "Signup failed");
      }
    } catch (error) {
      showMessage("error", "An error occurred. Please try again.");
    }
  };

  return (
    <Section className="pt-24 pb-12">
      <div className="container mx-auto max-w-md relative">
        <Gradient />
        <div className="relative z-1 bg-n-8 border border-n-6 rounded-2xl p-8">
          <h2 className="text-3xl font-bold text-center text-n-1 mb-8">
            Create a NeuroFetch Account
          </h2>

          {message.text && (
            <div
              className={`mb-4 text-center p-3 rounded-lg ${
                message.type === "success"
                  ? "bg-green-500/20 text-green-300"
                  : "bg-red-500/20 text-red-300"
              }`}
            >
              {message.text}
            </div>
          )}

          <form onSubmit={handleSignup}>
            <div className="mb-6">
              <label
                htmlFor="signupEmail"
                className="block mb-2 text-sm font-medium text-n-2"
              >
                Email
              </label>
              <input
                type="email"
                id="signupEmail"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-3 bg-n-7 border border-n-5 rounded-lg text-n-1 focus:ring-accent-1 focus:border-accent-1"
                required
              />
            </div>
            <div className="mb-6">
              <label
                htmlFor="signupPassword"
                className="block mb-2 text-sm font-medium text-n-2"
              >
                Password
              </label>
              <input
                type="password"
                id="signupPassword"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full p-3 bg-n-7 border border-n-5 rounded-lg text-n-1 focus:ring-accent-1 focus:border-accent-1"
                required
              />
            </div>
            <div className="mb-6">
              <label
                htmlFor="confirmPassword"
                className="block mb-2 text-sm font-medium text-n-2"
              >
                Confirm Password
              </label>
              <input
                type="password"
                id="confirmPassword"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full p-3 bg-n-7 border border-n-5 rounded-lg text-n-1 focus:ring-accent-1 focus:border-accent-1"
                required
              />
            </div>
            <Button type="submit" className="w-full">
              Sign Up
            </Button>
          </form>

          <p className="mt-8 text-center text-n-4">
            Already have an account?{" "}
            <Link to="/login" className="text-accent-1 hover:underline">
              Login
            </Link>
          </p>
        </div>
      </div>
    </Section>
  );
};

export default Signup; 