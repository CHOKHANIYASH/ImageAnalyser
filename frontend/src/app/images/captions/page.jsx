"use client";
import Image from "next/image";
import React from "react";
import { useState, useEffect } from "react";
import { CardBody, CardContainer, CardItem } from "../../../components/ui/card";
import Link from "next/link";
import axios from "axios";
import { useSearchParams } from "next/navigation";
import { useAppSelector } from "@/redux/hooks/index";
export default function Caption() {
  const searchParams = useSearchParams();
  const imageUrl = searchParams.get("imageUrl");
  const image = imageUrl.split("/").pop();
  const [caption, setCaption] = useState("");
  const accessToken = useAppSelector((state) => state.accessToken);
  useEffect(() => {
    axios
      .post(
        `${process.env.NEXT_PUBLIC_SERVER_MODEL_URL}/predict/caption?image_url=${imageUrl}`
      )
      .then((response) => {
        const originalCaption = response.data.caption;
        const filteredCaption = originalCaption
          .replace(/\bstartseq\b/gi, "") // Remove "start" (case-insensitive)
          .replace(/\bendseq\b/gi, "") // Remove "end" (case-insensitive)
          .trim(); // Remove any extra spaces
        setCaption(filteredCaption);
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
          {caption !== "" ? (
            caption
          ) : (
            <h1 className="">No Caption Available</h1>
          )}
        </div>
      </div>
    </>
  );
}
