"use client";
import Image from "next/image";
import React from "react";
import { useState, useEffect } from "react";
import { CardBody, CardContainer, CardItem } from "../../../components/ui/card";
import axios from "axios";
import { useSearchParams } from "next/navigation";
import { useAppSelector } from "@/redux/hooks/index";
// import { headers } from "next/headers";
export default function Tag() {
  const searchParams = useSearchParams();
  const imageUrl = searchParams.get("imageUrl");
  // const image = imageUrl.split("/").pop();
  const [tags, setTags] = useState([]);
  // const accessToken = useAppSelector((state) => state.accessToken);
  useEffect(() => {
    axios
      .post(
        // `${process.env.NEXT_PUBLIC_SERVER_MODEL_URL}/predict/tags?image_url=${imageUrl}`
        `${process.env.NEXT_PUBLIC_SERVER_MODEL_URL}/predict/test?image_url=${imageUrl}`
      )
      .then((response) => {
        console.log(response.data.Tags);
        setTags(response.data.Tags);
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);
  return (
    <>
      <div className="flex flex-col items-center justify-center gap-20 mt-10 md:flex-row">
        <Image
          src={imageUrl}
          height="400"
          width="400"
          className="object-cover max-sm:p-2 max-sm:rounded-3xl rounded-xl group-hover/card:shadow-xl"
          alt="thumbnail"
        />
        <div className="flex flex-col items-center text-2xl font-bold text-neutral-800">
          {tags !== "" && tags.length > 0 ? (
            <ul className="ml-5 list-disc">
              {tags.map((tag) => (
                <li key={tag} className="m-5">
                  {tag}
                </li>
              ))}
            </ul>
          ) : (
            <h1>No Tags Available</h1>
          )}
        </div>
      </div>
    </>
  );
}
