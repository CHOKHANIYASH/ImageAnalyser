import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "./Navbar";
const inter = Inter({ subsets: ["latin"] });
import "react-toastify/dist/ReactToastify.css";
import { ToastContainer } from "react-toastify";

import ReduxProvider from "./ReduxProvider";
export const metadata = {
  title: "Image Analyzer",
  description: "",
};
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-cyan-600`}>
        <ReduxProvider>
          <Navbar />
          <ToastContainer autoClose={2000} />
          {children}
        </ReduxProvider>
      </body>
    </html>
  );
}
