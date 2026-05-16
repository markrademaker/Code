export type MonthClimate = {
  month: string;
  highC: number;
  lowC: number;
  rainMm: number;
  seaC: number;
  blurb: string;
};

// Approximate monthly climate normals for Platja d'Aro (Costa Brava).
// Source: long-term Mediterranean coastal averages — treat as a guide,
// not a forecast.
export const PLATJA_CLIMATE: MonthClimate[] = [
  { month: "January",   highC: 13, lowC: 5,  rainMm: 50, seaC: 13, blurb: "Cool and crisp, quiet beaches." },
  { month: "February",  highC: 14, lowC: 6,  rainMm: 40, seaC: 13, blurb: "Mild winter days, occasional rain." },
  { month: "March",     highC: 16, lowC: 7,  rainMm: 50, seaC: 13, blurb: "Spring sets in, almond blossom." },
  { month: "April",     highC: 18, lowC: 9,  rainMm: 55, seaC: 14, blurb: "Pleasant for hiking and cycling." },
  { month: "May",       highC: 21, lowC: 12, rainMm: 50, seaC: 16, blurb: "Warm afternoons, sea still fresh." },
  { month: "June",      highC: 25, lowC: 16, rainMm: 35, seaC: 20, blurb: "Long sunny days, lively coves." },
  { month: "July",      highC: 28, lowC: 19, rainMm: 25, seaC: 23, blurb: "Peak summer, warm sea, busy beaches." },
  { month: "August",    highC: 28, lowC: 19, rainMm: 50, seaC: 25, blurb: "Hot, warmest sea, occasional storms." },
  { month: "September", highC: 25, lowC: 17, rainMm: 80, seaC: 23, blurb: "Warm sea, fewer crowds." },
  { month: "October",   highC: 21, lowC: 13, rainMm: 95, seaC: 20, blurb: "Mellow autumn, still swimmable." },
  { month: "November",  highC: 16, lowC: 9,  rainMm: 75, seaC: 17, blurb: "Cooler, atmospheric off-season." },
  { month: "December",  highC: 13, lowC: 6,  rainMm: 60, seaC: 14, blurb: "Quiet, festive in the old town." },
];
