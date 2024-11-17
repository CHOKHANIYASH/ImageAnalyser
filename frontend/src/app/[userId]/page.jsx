"use client";
import Image from "next/image";
import React, { act } from "react";
import { useState } from "react";
import { useParams } from "next/navigation";
import { useRouter } from "next/navigation";
import { CardBody, CardContainer, CardItem } from "../../components/ui/card";
import axios from "axios";
import Loader from "react-js-loader";
import ProtectedRoute from "../protectedRoute";
import { useAppSelector, useAppDispatch } from "@/redux/hooks/index";
import { toast } from "react-toastify";
function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploaded, setIsUploaded] = useState(false);
  const params = useParams();
  const router = useRouter();
  const url = process.env.NEXT_PUBLIC_SERVER_DEV_URL;
  const userId = params.userId;
  const dispatch = useAppDispatch();
  const accessToken = useAppSelector((state) => state.accessToken);

  const handleSubmit = async (e, action) => {
    try {
      e.preventDefault();
      console.log(selectedFile);
      if (selectedFile === null) {
        const file = { name: "Please select a file first" };
        setSelectedFile(file);
      } else {
        setIsUploaded(true);
        const uploadResponse = await axios.post(
          `${url}/images/upload/${userId}`,
          { images: selectedFile },
          {
            headers: {
              "Content-Type": "multipart/form-data",
              access_token: `${accessToken}`,
            },
          }
        );
        const { imageUrl } = uploadResponse.data.data;
        // console.log(action);
        if (action === "tag") {
          router.push(`/images/tags?imageUrl=${imageUrl}`);
        } else if (action === "caption") {
          router.push(`/images/captions?imageUrl=${imageUrl}`);
        }
      }
    } catch (err) {
      setIsUploaded(false);
      toast.error(
        "An error occured while uploading the image, Pls try again ",
        { toastId: "uniqueToastHome" }
      );
      console.log(err);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  const isValidFileType = (file) => {
    const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
    return allowedTypes.includes(file.type);
  };
  return (
    <>
      <form className="flex items-center justify-center w-full mt-20 h-96">
        <CardContainer className="w-1/2 ml-5 inter-var h-2/3 ">
          <CardBody className="bg-gray-50 relative group/card border-black/[0.1]  rounded-xl p-6 border w-full flex flex-col justify-center items-center">
            <CardItem translateZ="100" className="mt-4">
              <input
                type="file"
                id="imageUpload"
                onChange={handleImageChange}
                accept=".jpg,.jpeg,.png"
                style={{ display: "none" }}
              />
              <label htmlFor="imageUpload">
                <Image
                  src="/upload.svg"
                  height="100"
                  width="100"
                  className="object-cover cursor-pointer rounded-xl group-hover/card:shadow-xl"
                  alt="thumbnail"
                />
              </label>
            </CardItem>
            {selectedFile && (
              <p className="mt-4 text-lg font-bold text-neutral-600">
                {selectedFile.name}
              </p>
            )}
            {!isUploaded ? (
              <div className="flex justify-between w-full">
                <button
                  className="px-4 py-2 mt-10 text-sm font-bold text-white bg-black rounded-xl"
                  type="submit"
                  onClick={(e) => handleSubmit(e, "tag")}
                >
                  Tag
                </button>
                <button
                  className="px-4 py-2 mt-10 text-sm font-bold text-white bg-black rounded-xl"
                  type="submit"
                  onClick={(e) => handleSubmit(e, "caption")}
                >
                  Caption
                </button>
              </div>
            ) : (
              <Loader type="spinner-circle" bgColor={"#000000"} size={50} />
            )}
          </CardBody>
        </CardContainer>
      </form>
    </>
  );
}

export default ProtectedRoute(Home);
