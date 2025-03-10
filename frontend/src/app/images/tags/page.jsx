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
  const [tags, setTags] = useState("");
  // const accessToken = useAppSelector((state) => state.accessToken);
  console.log(
    "url ",
    `https://d7i9m4qiyzzde.cloudfront.net/predict/tags?image_url=${imageUrl}`
  );
  useEffect(() => {
    axios
      .post(
        `https://d7i9m4qiyzzde.cloudfront.net/predict/tags?image_url=${imageUrl}`
      )
      .then((response) => {
        setTags(response.data.Tag);
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
          {tags !== "" ? tags : <h1 className="">No Tags Available</h1>}
        </div>
      </div>
    </>
  );
}
