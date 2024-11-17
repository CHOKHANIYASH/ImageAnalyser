"use client";
import Image from "next/image";
import React from "react";
import { useState } from "react";
import { CardBody, CardContainer, CardItem } from "../../../components/ui/card";
import Link from "next/link";

export default function Caption() {
  const searchParams = useSearchParams();
  const imageUrl = searchParams.get("imageUrl");
  const image = imageUrl.split("/").pop();
  const [caption, setCaption] = useState([]);
  const accessToken = useAppSelector((state) => state.accessToken);
  useEffect(() => {
    axios
      .get(`${process.env.NEXT_PUBLIC_SERVER_MODEL_URL}/?image_url=${image}`, {
        headers: {
          access_token: `${accessToken}`,
        },
      })
      .then((response) => {
        setCaption(response.caption);
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
        <div className="flex flex-col items-center">
          {caption.length !== 0 ? (
            { caption }
          ) : (
            <h1 className="text-2xl font-bold text-neutral-800">
              No Tags Available
            </h1>
          )}
        </div>
      </div>
    </>
  );
}
